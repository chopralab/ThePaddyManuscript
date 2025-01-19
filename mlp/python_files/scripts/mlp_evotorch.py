import argparse
import torch
import time
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation
import os
import pandas as pd
import sys

# Argument parsing for repeats and generations
parser = argparse.ArgumentParser(description="MLP optimization with EvoTorch")
parser.add_argument("--generations", type=int, default=7, help="Number of generations per repeat")
parser.add_argument("--repeats", type=int, default=100, help="Number of repeats for optimization")
args = parser.parse_args()
generations = args.generations
repeats = args.repeats

# Set random seed for reproducibility
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get SLURM job ID if available
job_id = os.getenv("SLURM_JOB_ID", "unknown")
summary_log_filename = f'mlp_evotorch_summary_{job_id}.log'

# Log system information
print("Checking for GPU availability...")
with open(summary_log_filename, 'a') as log_file:
    log_file.write(f"PyTorch version: {torch.__version__}\n")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        log_file.write(f"GPU available: {gpu_name}\n")
        log_file.write(f"CUDA version: {torch.version.cuda}\n")
        print(f"GPU available: {gpu_name}")
    else:
        log_file.write("No GPU available, using CPU.\n")
        print("No GPU available, using CPU.")

# Load and preprocess dataset
print("Loading dataset...")
try:
    dataframe = pd.read_csv("d4990.csv", header=None)
    dataset = dataframe.values
    X = dataset[:, 1:].astype(np.float32)
    Y = dataset[:, 0]
    
    # Encode labels
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(encoded_Y)
    
    print("Dataset loaded and preprocessed successfully.")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    sys.exit(1)

# Define updated PyTorch model to match Paddy implementation
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout1, dropout2):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout2)
        )
        self.output = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.output(out)
        return out

# Define evaluation function for EvoTorch
def mlp_eval_func(hyperparameters):
    start_time = time.time()
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=4)
    val_scores = []
    
    # Extract hyperparameters
    dropout1 = float(hyperparameters[0])
    hidden_size1 = int(hyperparameters[1])
    dropout2 = float(hyperparameters[2])
    hidden_size2 = int(hyperparameters[3])

    for fold, (train_index, val_index) in enumerate(kfold.split(X.numpy(), encoded_Y)):
        # Prepare data for current fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False)

        # Initialize model and optimizer
        model = NeuralNet(
            input_size=2048,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            num_classes=30,
            dropout1=dropout1,
            dropout2=dropout2
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), eps=1e-07)

        # Training loop
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            # Validation
            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
                    val_batch_count += 1
            
            val_loss /= val_batch_count
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Final validation score
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
        
        val_score = f1_score(encoded_Y[val_index], val_preds, average='micro')
        val_scores.append(val_score)

    avg_score = sum(val_scores) / len(val_scores)
    elapsed_time = time.time() - start_time
    repeat_number = mlp_eval_func.run_counter
    mlp_eval_func.run_counter += 1
    
    print(f"Repeat {repeat_number} completed - Average F1 Score: {avg_score:.4f}, Time: {elapsed_time:.2f} seconds")
    with open(summary_log_filename, 'a') as log_file:
        log_file.write(f"Repeat {repeat_number} completed - Average F1 Score: {avg_score:.4f}, Time: {elapsed_time:.2f} seconds\n")
    
    return torch.tensor(avg_score)

# Initialize run counter
mlp_eval_func.run_counter = 1

# Main optimization loop
for repeat in range(repeats):
    print(f"\nStarting optimization repeat {repeat + 1}/{repeats}")
    
    # Set up the optimization problem with matching parameter ranges from Paddy
    problem = Problem(
        "max",
        mlp_eval_func,
        solution_length=4,
        initial_bounds=([0.0, 500, 0.0, 32], [0.5, 1000, 0.5, 500]),
        bounds=([0.0, 300, 0.0, 30], [1.0, 3000, 1.0, 2000]),
        dtype=torch.float32,
    )

    # Set up the genetic algorithm
    searcher = GeneticAlgorithm(
        problem,
        popsize=20,
        operators=[GaussianMutation(problem, stdev=0.2)]
    )

    StdOutLogger(searcher, interval=30)
    pandas_logger = PandasLogger(searcher)

    # Run optimization
    try:
        for generation in range(generations):
            searcher.step()
            best_solution_gen = searcher.status["best"]
            best_value_gen = best_solution_gen.evaluation.item()
            print(f"Generation {generation + 1} - Best F1 Score: {best_value_gen:.4f}")
            with open(summary_log_filename, 'a') as log_file:
                log_file.write(f"Generation {generation + 1} - Best F1 Score: {best_value_gen:.4f}\n")
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)

# Calculate and log final summary
print("\nCalculating summary of all repeats...")
with open(summary_log_filename, 'r') as log_file:
    scores = [float(line.split('Average F1 Score: ')[1].split(',')[0]) 
             for line in log_file if 'Repeat' in line]
    times = [float(line.split('Time: ')[1].split(' ')[0]) 
            for line in log_file if 'Repeat' in line]
    
    summary_stats = {
        'Best F1 Score': max(scores),
        'Worst F1 Score': min(scores),
        'Average F1 Score': sum(scores) / len(scores),
        'Average Runtime': sum(times) / len(times)
    }

    with open(summary_log_filename, 'a') as log_file_append:
        log_file_append.write("\nFinal Summary:\n")
        for metric, value in summary_stats.items():
            log_file_append.write(f"- {metric}: {value:.4f}\n")

print("Optimization completed successfully.")