#!/bin/bash
#SBATCH --job-name=evotorch_optimization_11Dec_200pop_EA  # Job name
#SBATCH --output=Evotorch_log_11dec/%x_%j.log  # Output log file (%x is job name, %j is job ID)
#SBATCH --error=Evotorch_log_11dec/%x_%j.err   # Error log file
#SBATCH --partition=gilbreth-k  # Partition name
#SBATCH --account=gchopra-k     # Account name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=14-00:00:00      # Maximum run time
#SBATCH --mem=16GB              # Memory per node

# Create log directory if it doesn't exist
mkdir -p Evotorch_log_11dec

# Load your required modules or conda environment
module load anaconda/2020.11
conda activate /scratch/gilbreth/iyer95/U18  # Full path to U18 environment

# Run the EvoTorch Python script with user-specified generations, repeats, and std_dev
python minmax_evotorch.py --generations 6 --repeats 100 --std_dev 0.2 --job_name "$SLURM_JOB_ID" --seed 42