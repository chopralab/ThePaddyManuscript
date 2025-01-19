import random
import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ax import OptimizationLoop, ParameterType, RangeParameter, SearchSpace
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import logging

# ============================
# Set Random Seeds for Reproducibility
# ============================
seed = 7  # Consistent seed value
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================
# Configure Logging
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# Device Configuration
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================
# Load Data
# ============================
logger.info("Loading data...")
dataframe = pd.read_csv("d4990.csv", header=None)
dataset = dataframe.values
X = dataset[:, 1:].astype(np.float32)  # Convert to float32 for PyTorch compatibility
Y = dataset[:, 0]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
y = torch.tensor(encoded_Y, dtype=torch.long)
logger.info(f"Data loading complete. Shape - X: {X.shape}, Y: {y.shape}")

# ============================
# Define the Neural Network
# ============================
class MLPModel(nn.Module):
    def __init__(self, input_dim, leng_1, dropout_1, leng_2, dropout_2, output_dim):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, leng_1),
            nn.ReLU(),
            nn.Dropout(dropout_1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(leng_1, leng_2),
            nn.ReLU(),
            nn.Dropout(dropout_2)
        )
        self.output_layer = nn.Linear(leng_2, output_dim)  # Output classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

# ============================
# Evaluation Function
# ============================
def evaluate_model(params, trial_number, iteration_number):
    logger.info(f"\nRunning Ax optimization trial {trial_number} (Iteration {iteration_number} of {total_iterations})")
    
    # Log GPU memory usage before training
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"GPU {i} - Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")
    
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, encoded_Y), start=1):
        logger.info(f"Starting Fold {fold} of 3...")
        
        # Create training and validation tensors
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = y[train_idx]
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = y[val_idx]
        
        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        batch_size = 1000  # Fixed batch size
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")
        logger.info(f"Using batch size: {batch_size}")
        
        # Initialize model, criterion, optimizer
        model = MLPModel(
            input_dim=X.shape[1],
            leng_1=int(params['leng_1']),
            dropout_1=params['dropout_1'],
            leng_2=int(params['leng_2']),
            dropout_2=params['dropout_2'],
            output_dim=len(np.unique(encoded_Y))
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07)  # Explicit eps
        
        # Training loop with DataLoader
        model.train()
        for epoch in range(5):  # 5 epochs
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
        
        f1 = f1_score(encoded_Y[val_idx], val_preds, average='micro')
        f1_scores.append(f1)
        
        logger.info(f"Fold {fold} complete - F1 Score: {f1:.4f}")
        torch.cuda.empty_cache()  # Clear unused memory after each fold
    
    avg_f1 = np.mean(f1_scores)
    logger.info(f"Trial {trial_number} complete - Average F1 Score: {avg_f1:.4f}")
    return avg_f1

# ============================
# Checkpoint Functions
# ============================
def save_checkpoint(results, counter, f1_scores, run_times, job_id, directory="checkpoints"):
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(directory, f"checkpoint_{job_id}.json")  # Use job_id in filename
    checkpoint_data = {
        "results": results,
        "counter": counter,
        "f1_scores": f1_scores,
        "run_times": run_times
    }
    with open(file_path, "w") as f:
        json.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved at iteration {counter + 1} to {file_path}.")

def load_checkpoint(job_id, directory="checkpoints"):
    file_path = os.path.join(directory, f"checkpoint_{job_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            checkpoint_data = json.load(f)
        logger.info(f"Checkpoint loaded from {file_path}.")
        return (
            checkpoint_data["results"],
            checkpoint_data["counter"],
            checkpoint_data["f1_scores"],
            checkpoint_data["run_times"]
        )
    else:
        return [], 0, [], []

# ============================
# Main Optimization Setup with Custom Generation Strategy
# ============================

# Load job ID from SLURM environment variable (or use a default for testing)
job_id = os.getenv("SLURM_JOB_ID", "default")

# Load checkpoint if exists
results, counter, f1_scores, run_times = load_checkpoint(job_id)

# Define total iterations
total_iterations = 150  # 25 Sobol + 125 GPEI

# Define Ax optimization parameters
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="leng_1", parameter_type=ParameterType.INT, lower=300, upper=3000),
        RangeParameter(name="dropout_1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0),
        RangeParameter(name="leng_2", parameter_type=ParameterType.INT, lower=32, upper=2000),
        RangeParameter(name="dropout_2", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0),
    ]
)

# Define the custom Generation Strategy
generation_strategy = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,  # Initial Sobol generator
            num_trials=25,
            min_trials_observed=25,
        ),
        GenerationStep(
            model=Models.GPEI,  # Followed by GPEI
            num_trials=125,
        ),
    ]
)

# Initialize the Optimization Loop
optimization_loop = OptimizationLoop(
    search_space=search_space,
    evaluation_function=lambda params: evaluate_model(params, trial_number=len(results)+1, iteration_number=len(results)+1),
    objective_name='f1',
    minimize=False,
    generation_strategy=generation_strategy,
)

# ============================
# Main Optimization Loop with Generation Strategy
# ============================

# Start the optimization loop with progress tracking
for counter in tqdm(range(counter, total_iterations), desc="Ax Optimization Iterations"):
    logger.info(f"\nStarting Ax optimization iteration {counter + 1} of {total_iterations}...")
    start_time = time.time()
    
    # Generate a new trial
    generator_run = optimization_loop.gen(n=1)
    trial = generator_run.trials[0]
    parameters = trial.arm.parameters
    trial_index = trial.index
    
    logger.info(f"Generated Trial {trial_number := len(results)+1} with parameters: {parameters}")
    
    # Evaluate the trial
    f1_score_value = optimization_loop.evaluation_function(parameters)
    
    # Complete the trial with the obtained F1 score
    optimization_loop.complete_trial(trial_index, f1_score_value)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    run_times.append(elapsed_time)
    
    # Safely access the F1 score value with a try-except block
    try:
        f1_score_recorded = f1_score_value
    except (TypeError, AttributeError) as e:
        logger.error(f"Error accessing 'f1' value: {e}")
        f1_score_recorded = None
    
    if f1_score_recorded is not None:
        f1_scores.append(f1_score_recorded)
    else:
        logger.warning("f1_score_value is None; skipping this iteration's result.")
    
    logger.info(f"Iteration {counter + 1} completed. Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"Best parameters so far: {optimization_loop.generation_strategy._best.parameters}")
    if optimization_loop.best_result() is not None:
        logger.info(f"Best value (F1 Score) so far: {optimization_loop.best_result().objective_mean:.4f}")
    else:
        logger.info("Best value (F1 Score) so far: N/A")
    
    results.append({
        "iteration": counter + 1,
        "best_parameters": optimization_loop.generation_strategy._best.parameters if optimization_loop.generation_strategy._best else None,
        "best_value": f1_score_recorded,
        "time_taken": elapsed_time,
    })
    
    # Save checkpoint after each iteration
    save_checkpoint(results, counter, f1_scores, run_times, job_id)

# ============================
# Summary of Results
# ============================
logger.info("\nOptimization complete for all iterations.")
logger.info("F1 Score Summary for Each Iteration:")
for result in results:
    if result["best_value"] is not None:
        logger.info(f"Iteration {result['iteration']}: Best F1 Score: {result['best_value']:.4f}")
    else:
        logger.info(f"Iteration {result['iteration']}: Best F1 Score: N/A")
    
# Calculate the overall best, worst, and average F1 score and runtime
valid_f1_scores = [score for score in f1_scores if score is not None]
if valid_f1_scores:
    best_f1_score = max(valid_f1_scores)
    worst_f1_score = min(valid_f1_scores)
    average_f1_score = np.mean(valid_f1_scores)
else:
    best_f1_score = worst_f1_score = average_f1_score = None

average_runtime = np.mean(run_times) if run_times else None

logger.info("\nOverall Summary:")
if best_f1_score is not None:
    logger.info(f"Best F1 Score across all iterations: {best_f1_score:.4f}")
    logger.info(f"Worst F1 Score across all iterations: {worst_f1_score:.4f}")
    logger.info(f"Average F1 Score across all iterations: {average_f1_score:.4f}")
else:
    logger.info("No valid F1 Scores were recorded.")
if average_runtime is not None:
    logger.info(f"Average runtime per iteration: {average_runtime:.2f} seconds")
else:
    logger.info("No runtime data available.")

# Print summary
print("\nSummary of all F1 Scores and Runtimes:")
if best_f1_score is not None:
    print(f"Best F1 Score: {best_f1_score:.4f}")
    print(f"Worst F1 Score: {worst_f1_score:.4f}")
    print(f"Average F1 Score: {average_f1_score:.4f}")
else:
    print("Best F1 Score: N/A")
    print("Worst F1 Score: N/A")
    print("Average F1 Score: N/A")

if average_runtime is not None:
    print(f"Average runtime: {average_runtime:.2f} seconds")
else:
    print("Average runtime: N/A")
