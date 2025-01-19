#!/bin/bash
#SBATCH --job-name=gramacy_Hyperopt_11Jan25  # Job name
#SBATCH --output=GRAMACY_K_11Jan/%x_%j.log                    # Output log file (%x is job name, %j is job ID)
#SBATCH --error=GRAMACY_K_11Jan/%x_%j.err                     # Error log file
#SBATCH --partition=gilbreth-k  # Partition name
#SBATCH --account=gchopra-k     # Account name
#SBATCH --constraint=K                                         # Enforce node type K
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=14-00:00:00      # Maximum run time
#SBATCH --mem=8GB              # Memory per node

mkdir -p GRAMACY_K_11Jan
mkdir -p GRAMACY_K_11Jan/pickle

# Load required modules
module load anaconda/2020.11
source activate /scratch/gilbreth/iyer95/U18

# Define the save directory
SAVE_DIR="/scratch/gilbreth/iyer95/pad_clean/gramacy/GRAMACY_K_11Jan/pickle/"

# Run the optimization script with all arguments
python gramacy_hyperopt_DEBUG.py \
    --save_directory "$SAVE_DIR" \
    --job_name "$SLURM_JOB_ID" \
    --seed 42 \
    --num_iterations 100 \
    --max_evals 1500 \


# (Optional) If you'd like to extract any summary information after completion, you can still do so here.
# Example:
# grep -E "Running optimization trial|parameterization:|objective value:" \
#    DEBUG_GRAMACY_K_11Jan/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log \
#    > DEBUG_GRAMACY_K_11Jan/${SLURM_JOB_NAME}_summary_${SLURM_JOB_ID}.log
