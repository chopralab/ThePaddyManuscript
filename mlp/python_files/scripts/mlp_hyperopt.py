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
from hyperopt import fmin, tpe, hp, Trials

# Set random seed for reproducibility
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
print("\nLoading dataset...")
dataframe = pd.read_csv("d4990.csv", header=None)
dataset = dataframe.values
X = dataset[:, 1:].astype(np.float32)  # Changed to float32 for compatibility with PyTorch
Y = dataset[:, 0]
print(f"Dataset loaded. Shape - X: {X.shape}, Y: {Y.shape}")

# Encode labels
print("\nEncoding labels...")
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
print(f"Unique classes: {len(np.unique(encoded_Y))}")

# Convert labels to tensor
y = torch.tensor(encoded_Y, dtype=torch.long)

# Define the search space for hyperparameters
fspace = {
    'leng_1': hp.quniform('leng_1', 300, 3000, 1),
    'dropout_1': hp.uniform('dropout_1', 0, 1),
    'leng_2': hp.quniform('leng_2', 32, 2000, 1),
    'dropout_2': hp.uniform('dropout_2', 0, 1)
}

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

def create_model(values):
    model = MLPModel(
        input_dim=X.shape[1],
        leng_1=int(values['leng_1']),
        dropout_1=values['dropout_1'],
        leng_2=int(values['leng_2']),
        dropout_2=values['dropout_2'],
        output_dim=len(np.unique(encoded_Y))
    )
    return model.to(device)

def run_func(values):
    p_start = time.time()
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=4)
    validation_f1_scores = []
    
    print("\n" + "="*50)
    print(f"Evaluating model with hyperparameters:")
    print(f"Layer 1 size: {int(values['leng_1'])}")
    print(f"Layer 1 dropout: {values['dropout_1']:.3f}")
    print(f"Layer 2 size: {int(values['leng_2'])}")
    print(f"Layer 2 dropout: {values['dropout_2']:.3f}")
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X, encoded_Y), 1):
        print(f"\nTraining Fold {fold}/3:")
        
        # Create training and validation tensors
        X_train = torch.tensor(X[train_index], dtype=torch.float32)
        y_train = y[train_index]
        X_val = torch.tensor(X[val_index], dtype=torch.float32)
        y_val = y[val_index]
        
        # Create TensorDataset and DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        batch_size = 1000  # You can adjust this value or make it a hyperparameter
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Using batch size: {batch_size}")
        
        # Initialize model, criterion, optimizer
        model = create_model(values)
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-07)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with DataLoader
        model.train()
        for epoch in range(5):
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
            print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Validation predictions
            val_preds = []
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
            
            # Training predictions (optional, can be removed if not needed)
            train_preds = []
            for batch_X, _ in train_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
        
        val_f1 = f1_score(encoded_Y[val_index], val_preds, average='micro')
        train_f1 = f1_score(encoded_Y[train_index], train_preds, average='micro')
        
        validation_f1_scores.append(val_f1)
        
        print(f"Fold {fold} Results:")
        print(f"Validation F1 Score: {val_f1:.4f}")
    
    avg_val_f1 = sum(validation_f1_scores) / len(validation_f1_scores)
    
    print("\nOverall Results:")
    print(f"Average Validation F1 Score: {avg_val_f1:.4f}")
    print(f"Time taken: {time.time() - p_start:.2f} seconds")
    print("="*50)
    
    return -avg_val_f1  # Return negative because hyperopt minimizes

# Initialize the RNG once, outside the loop
rng = np.random.default_rng(seed)

# Hyperparameter optimization loop
bs_counter = 0
while bs_counter < 100:
    print(f"\nStarting optimization iteration {bs_counter + 1}/100")
    trials = Trials()
    start = time.time()
    
    best = fmin(
        fn=run_func,
        space=fspace,
        algo=tpe.suggest,
        max_evals=150,
        trials=trials,
        rstate=rng  # Use the pre-initialized RNG
    )
    
    end = time.time()
    print(f"\nOptimization Iteration {bs_counter + 1} Complete:")
    print(f"Best hyperparameters found:")
    print(f"Layer 1 size: {int(best['leng_1'])}")
    print(f"Layer 1 dropout: {best['dropout_1']:.3f}")
    print(f"Layer 2 size: {int(best['leng_2'])}")
    print(f"Layer 2 dropout: {best['dropout_2']:.3f}")
    print(f"Best validation F1 score: {-min(trials.losses()):.4f}")
    print(f"Time taken: {end - start:.2f} seconds")
    print("-"*50)
    bs_counter += 1
