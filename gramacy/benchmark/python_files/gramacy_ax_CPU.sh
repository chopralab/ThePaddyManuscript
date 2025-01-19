#!/bin/bash
#SBATCH --job-name=ax_gramacy_12Dec24__500trialsH_CPUparallel # Job name
#SBATCH --output=ax_12dec/%x_%j.log                    # Output log file (%x is job name, %j is job ID)
#SBATCH --error=ax_12dec/%x_%j.err                     # Error log file
#SBATCH --partition=gilbreth-h  # Partition name
#SBATCH --account=gchopra-h     # Account name
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks=1                            # Number of tasks
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:1                          # Number of GPUs per node
#SBATCH --time=14-00:00:00                     # Maximum run time (6 days)
#SBATCH --mem=8GB                             # Memory per node

# Create output directory if it doesn't exist
mkdir -p ax_12dec

# Load required modules
module load anaconda/2020.11
source activate /scratch/gilbreth/iyer95/U18

# Define the save directory
SAVE_DIR="/scratch/gilbreth/iyer95/pad/gramacy/results/ax_12dec/"

# Run the optimization script with the save_directory and job_name arguments
python gramacy_ax_CPU.py --save_directory "$SAVE_DIR" --job_name "$SLURM_JOB_ID" >> ax_12dec/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1 --seed 42 --num_iterations 100 --total_trials 500 --sobol_num_trials 25 --gpei_num_trials 475 --parallel_workers 10

# Extract summary information
grep -E "Running optimization trial|parameterization:|objective value:" ax_12dec/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log > ax_12dec/${SLURM_JOB_NAME}_summary_${SLURM_JOB_ID}.log

