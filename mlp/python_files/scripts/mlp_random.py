import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os

# ==========================
# 1. Device Configuration
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# 2. Set Random Seeds
# ==========================
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)

# ==========================
# 3. CUDA Configuration
# ==========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure only GPU 0 is visible

# ==========================
# 4. Load and Prepare Dataset
# ==========================
print("\nLoading dataset...")
dataframe = pd.read_csv("d4990.csv", header=None)
dataset = dataframe.values
X = dataset[:, 1:].astype(np.float32)  # Changed to float32 for compatibility with PyTorch
Y = dataset[:, 0]
print(f"Dataset loaded. Shape - X: {X.shape}, Y: {Y.shape}")

# Encode target labels
print("\nEncoding labels...")
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
print(f"Unique classes: {len(np.unique(encoded_Y))}")

# Convert labels to tensor
y = torch.tensor(encoded_Y, dtype=torch.long)

# ==========================
# 5. Define MLP Model
# ==========================
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
        self.output = nn.Linear(leng_2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x

# ==========================
# 6. Hyperparameter Selection
# ==========================
def create_model():
    """
    Randomly selects hyperparameters and initializes the MLP model.
    Returns the model and the selected parameters.
    """
    l1 = np.random.randint(300, 3000)
    l2 = np.random.randint(32, 2000)
    d1 = np.random.uniform(0, 1)
    d2 = np.random.uniform(0, 1)
    model = MLPModel(input_dim=X.shape[1], leng_1=l1, dropout_1=d1, leng_2=l2, dropout_2=d2, output_dim=len(np.unique(encoded_Y)))
    params = {
        'layer1_units': l1,
        'layer2_units': l2,
        'dropout1_rate': round(d1, 3),
        'dropout2_rate': round(d2, 3)
    }
    return model, params

# ==========================
# 7. Training and Evaluation Function
# ==========================
def run_func():
    """
    Runs a single trial of training with randomly selected hyperparameters.
    Performs 3-fold cross-validation and returns the average F1 score along with the hyperparameters.
    """
    p_start = time.time()
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=4)
    fold_scores = []
    
    # Select hyperparameters and print model architecture
    model_params = create_model()
    model = model_params[0]  # Not used here; model will be re-initialized per fold
    params = model_params[1]
    print("\nModel Architecture:")
    print(f"Layer 1: {params['layer1_units']} units, dropout {params['dropout1_rate']}")
    print(f"Layer 2: {params['layer2_units']} units, dropout {params['dropout2_rate']}")

    # Iterate over each fold
    for fold, (train_index, val_index) in enumerate(kfold.split(X, encoded_Y), 1):
        print(f"\nTraining Fold {fold}/3:")
        
        # Prepare data for this fold
        X_train = torch.tensor(X[train_index], dtype=torch.float32)
        y_train = y[train_index]
        X_val = torch.tensor(X[val_index], dtype=torch.float32)
        y_val = y[val_index]
        
        # Create TensorDataset and DataLoader for training
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
        
        # Create TensorDataset and DataLoader for validation
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Using batch size: 1000")
        
        # Initialize a new model for this fold
        fold_model = MLPModel(
            input_dim=X.shape[1],
            leng_1=params['layer1_units'],
            dropout_1=params['dropout1_rate'],
            leng_2=params['layer2_units'],
            dropout_2=params['dropout2_rate'],
            output_dim=len(np.unique(encoded_Y))
        ).to(device)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(fold_model.parameters(), lr=0.001, eps=1e-07)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(5):
            fold_model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = fold_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
        
        # Evaluation
        fold_model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = fold_model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
        
        # Calculate validation F1 score
        fold_f1 = f1_score(encoded_Y[val_index], val_preds, average='micro')
        fold_scores.append(fold_f1)
        print(f"Fold {fold} Validation F1: {fold_f1:.4f}")
    
    # Compute average F1 score across all folds
    avg_val_f1 = sum(fold_scores) / len(fold_scores)
    elapsed_time = time.time() - p_start
    print(f"\nCross-Validation F1: {avg_val_f1:.4f}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    return avg_val_f1, params

# ==========================
# 8. Hyperparameter Optimization Loop
# ==========================
# Initialize variables to track the best overall performance
best_overall_f1 = 0
best_overall_params = None

for trial in range(100):  # 100 trials total
    print(f"\n{'='*50}")
    print(f"Trial {trial + 1}/100")
    print(f"{'='*50}")
    
    start = time.time()
    best_trial_f1 = 0
    best_trial_params = None
    
    for iteration in range(200):  # 200 iterations per trial
        print(f"\nIteration {iteration + 1}/200")
        current_f1, params = run_func()
        
        # Update the best score for this trial
        if current_f1 > best_trial_f1:
            best_trial_f1 = current_f1
            best_trial_params = params
            print("\n>>> New Best Score Found! <<<")
            print(f"F1 Score: {best_trial_f1:.4f}")
            print("Parameters:", best_trial_params)
        
        # Update the best overall score across all trials
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            best_overall_params = params
            print("\n>>> New Best Overall Score Found! <<<")
            print(f"F1 Score: {best_overall_f1:.4f}")
            print("Parameters:", best_overall_params)
    
    end = time.time()
    print(f"\nTrial {trial + 1} Summary:")
    print(f"Trial Time: {end - start:.2f} seconds")
    print(f"Best Trial F1: {best_trial_f1:.4f}")
    print("Best Parameters:", best_trial_params)

print("\nFinal Results:")
print(f"Best Overall F1: {best_overall_f1:.4f}")
print("Best Parameters:", best_overall_params)
