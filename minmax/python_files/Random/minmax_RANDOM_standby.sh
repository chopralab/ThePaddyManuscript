#!/bin/bash
#SBATCH --job-name=minmax_05Dec_RANDOM_STANDBY # Job name
#SBATCH --output=RANDOM_06dec/%x_%j.log  # Output log file (%x is job name, %j is job ID)
#SBATCH --error=RANDOM_06dec/%x_%j.err   # Error log file
#SBATCH --account=debug    # Account name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=30:00      # Maximum run time
#SBATCH --mem=16GB              # Memory per node

# Create the output directory if it doesn't exist
mkdir -p RANDOM_06dec

# Load your required modules or conda environment
module load anaconda/2020.11
conda activate /scratch/gilbreth/iyer95/U18  # Full path to U18 environment

# Run the Ax Python script with user-specified parameters and redirect output to log file
python minmax_RANDOM_standby.py --job_name "$SLURM_JOB_ID" >> RANDOM_06dec/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1

# Extract key information (parameters and scores) for summary
grep -E "Running optimization trial|parameterization:|objective value:" RANDOM_06dec/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log > RANDOM_06dec/${SLURM_JOB_NAME}_summary_${SLURM_JOB_ID}.log
