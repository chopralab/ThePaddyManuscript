#!/bin/bash
#SBATCH --job-name=Gramacy_Random_10Jan25 # Job name
#SBATCH --output=GRAMACY_K_10Jan/%x_%j.log                    # Output log file (%x is job name, %j is job ID)
#SBATCH --error=GRAMACY_K_10Jan/%x_%j.err                     # Error log file
#SBATCH --partition=gilbreth-k  # Partition name
#SBATCH --account=gchopra-k     # Account name
#SBATCH --constraint=K                                          # Enforce node type K
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=14-00:00:00      # Maximum run time
#SBATCH --mem=8GB              # Memory per node


# Create output directory if it doesn't exist
mkdir -p GRAMACY_K_10Jan

# Load required modules
module load anaconda/2020.11
source activate /scratch/gilbreth/iyer95/U18

# Define the save directory
SAVE_DIR="/scratch/gilbreth/iyer95/pad/gramacy/results/GRAMACY_K_10Jan/"

# Run the optimization script with the save_directory and job_name arguments
python gramacy_RANDOM.py --save_directory "$SAVE_DIR" --job_name "$SLURM_JOB_ID" --seed 42 >> GRAMACY_K_10Jan/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1 --seed 42

# Extract summary information
grep -E "Running optimization trial|parameterization:|objective value:" GRAMACY_K_10Jan/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log > GRAMACY_K_10Jan/${SLURM_JOB_NAME}_summary_${SLURM_JOB_ID}.log
