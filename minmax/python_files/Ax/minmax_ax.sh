#!/bin/bash
#SBATCH --job-name=minmax_02Dec_Ax_GenerationStrategy_500trialsH_PARALELL20x # Job name
#SBATCH --output=%x_%j.log  # Output log file (%x is job name, %j is job ID)
#SBATCH --error=%x_%j.err   # Error log file
#SBATCH --partition=gilbreth-h  # Partition name
#SBATCH --account=gchopra-h     # Account name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Number of GPUs per node
#SBATCH --time=6-00:00:00      # Maximum run time (14 days)
#SBATCH --mem=16GB              # Memory per node

# Load your required modules or conda environment
module load anaconda/2020.11
conda activate /scratch/gilbreth/iyer95/U18  # Full path to U18 environment

# Run the Ax Python script with user-specified parameters and redirect output to log file
python minmax_ax.py --job_name "$SLURM_JOB_ID" >> ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log 2>&1

# Extract key information (parameters and scores) for summary
grep -E "Running optimization trial|parameterization:|objective value:" ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log > ${SLURM_JOB_NAME}_summary_${SLURM_JOB_ID}.log