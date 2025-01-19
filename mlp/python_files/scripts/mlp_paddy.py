import itertools
import random
import pandas as pd 
import time
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import os
import sys
import paddy

print("Starting script execution...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set random seeds for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA seed set successfully")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout1, dropout2):
        super(NeuralNet, self).__init__()
        print(f"\nInitializing Neural Network with parameters:")
        print(f"Input size: {input_size}")
        print(f"Hidden size 1: {hidden_size1}")
        print(f"Hidden size 2: {hidden_size2}")
        print(f"Number of classes: {num_classes}")
        print(f"Dropout 1: {dropout1}")
        print(f"Dropout 2: {dropout2}")
        
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

def create_model(values):
    print("\nCreating model with values:", values)
    model = NeuralNet(
        input_size=2048,
        hidden_size1=int(values[1][0]),
        hidden_size2=int(values[3][0]),
        num_classes=30,
        dropout1=values[0][0],
        dropout2=values[2][0]
    ).to(device)
    print("Model created successfully")
    return model

def run_func(values):
    print("\n" + "="*50)
    print("Starting new run_func execution")
    start_p = time.time()
    
    try:
        print("\nLoading dataset...")
        dataframe = pd.read_csv("d4990.csv", header=None)
        dataset = dataframe.values
        X = dataset[:, 1:].astype(np.float32)
        Y = dataset[:, 0]
        print(f"Dataset loaded successfully. Shape - X: {X.shape}, Y: {Y.shape}")
        
        print("\nEncoding labels...")
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(Y)
        print(f"Unique classes: {len(np.unique(encoded_Y))}")
        
        print("\nConverting to PyTorch tensors...")
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(encoded_Y)
        print("Conversion successful")
        
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=4)
        val_scores = []
        train_scores = []
        
        print("\nStarting K-fold cross validation...")
        for fold, (train_index, val_index) in enumerate(kfold.split(X.numpy(), encoded_Y)):
            print(f"\nProcessing fold {fold+1}/3")
            
            # Create data loaders
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            
            print(f"Training set size: {len(X_train)}")
            print(f"Validation set size: {len(X_val)}")
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1000, shuffle=False)
            
            print("\nInitializing model and optimizer...")
            model = create_model(values)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), eps=1e-07)
            print("Initialization complete")
            
            patience = 5
            best_val_loss = float('inf')
            patience_counter = 0
            
            print("\nStarting training...")
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
                
                avg_epoch_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1}/5 - Average training loss: {avg_epoch_loss:.4f}")
                
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
                print(f"Validation loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print("New best validation loss!")
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
            
            print("\nCalculating metrics...")
            model.eval()
            with torch.no_grad():
                # Validation predictions
                val_preds = []
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                
                # Training predictions
                train_preds = []
                for batch_X, _ in train_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    train_preds.extend(predicted.cpu().numpy())
            
            val_score = f1_score(encoded_Y[val_index], val_preds, average='micro')
            train_score = f1_score(encoded_Y[train_index], train_preds, average='micro')
            
            print(f"Fold {fold+1} scores:")
            print(f"Validation F1: {val_score:.4f}")
            
            val_scores.append(val_score)
            train_scores.append(train_score)
        
        print("\nFinal scores across all folds:")
        print(f"Validation scores: {val_scores}")
        print(f"Average validation score: {sum(val_scores)/3:.4f}")
        print(f"Training scores: {train_scores}")
        print(f"Time elapsed: {time.time() - start_p:.2f} seconds")
        
        return sum(val_scores) / 3
    
    except Exception as e:
        print(f"\nERROR: An exception occurred:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise

# Define parameter space
print("\nDefining parameter space...")
dropout = paddy.PaddyParameter(param_range=[0, .5, .05], param_type='continuous', limits=[0, 1], gaussian='default', normalization=True)
layer1 = paddy.PaddyParameter(param_range=[500, 1000, 5], param_type='integer', limits=[300, 3000], gaussian='default', normalization=True)
layer2 = paddy.PaddyParameter(param_range=[32, 500, 5], param_type='integer', limits=[30, 2000], gaussian='default', normalization=True)

class space(object):
    def __init__(self):
        self.d1 = dropout
        self.l1 = layer1 
        self.d2 = dropout 
        self.l2 = layer2
        print("Parameter space initialized")

if __name__ == "__main__":
    try:
        print("\nStarting main execution...")
        test_space = space()
        bs_counter = 0
        
        print("\nBeginning optimization loop...")
        while bs_counter < 100:
            print(f"\nIteration {bs_counter+1}/100")
            start = time.time()
            
            runner = paddy.PFARunner(
                space=test_space,
                eval_func=run_func,
                paddy_type='generational',
                rand_seed_number=25,
                yt=5,
                Qmax=10,
                r=.2,
                iterations=7
            )
            
            print("Running PFARunner...")
            runner.run_paddy(verbose='all')
            
            iteration_time = time.time() - start
            print(f"Iteration {bs_counter+1} completed in {iteration_time:.2f} seconds")
            
            bs_counter += 1
            
    except Exception as e:
        print(f"\nFATAL ERROR in main execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise