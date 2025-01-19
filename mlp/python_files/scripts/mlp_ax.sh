#!/bin/bash
#SBATCH --job-name=mlp_AX_08Jan25_Paper_150iterations_Genstrategy # Job name for MLP optimization
#SBATCH --output=logs_AX_08jan25/%x_%j.log # Output log file (%x is job name, %j is job ID)
#SBATCH --error=logs_AX_08jan25/%x_%j.err # Error log file                  
#SBATCH --partition=gilbreth-k  # Partition name
#SBATCH --account=gchopra-k    # Account name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=14-00:00:00      # Maximum run time
#SBATCH --mem=64GB              # Memory per node

# Create logs directory if it doesn't exist
mkdir -p logs_AX_08jan25

# Load required modules
module load anaconda/2020.11
conda activate /scratch/gilbreth/iyer95/MLP  # Full path to U18 environment

# Run the EvoTorch Python script for MLP optimization with user-specified generations and repeats
python mlp_ax.py
