#!/bin/bash
#SBATCH --job-name=MLP_Random_10Jan25  # Job name
#SBATCH --output=MLP_K_10Jan/%x_%j.log                    # Output log file (%x is job name, %j is job ID)
#SBATCH --error=MLP_K_10Jan/%x_%j.err                     # Error log file
#SBATCH --partition=gilbreth-k  # Partition name
#SBATCH --account=gchopra-k     # Account name
#SBATCH --constraint=K                                          # Enforce node type K
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=14-00:00:00      # Maximum run time
#SBATCH --mem=64GB              # Memory per node

mkdir -p MLP_K_10Jan
# Load your required modules or conda environment
module load anaconda/2020.11
conda activate /scratch/gilbreth/iyer95/MLP  # Full path to U18 environment

# Run the EvoTorch Python script for MLP optimization with user-specified generations and repeats
python mlp_random.py
