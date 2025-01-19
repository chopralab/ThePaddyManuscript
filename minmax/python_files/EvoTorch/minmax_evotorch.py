import torch
import numpy as np
import random
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger, Logger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation
import argparse
import os

# Define the bimodal evaluation function
def bimodal_eval_func(input_tensor: torch.Tensor) -> torch.Tensor:
    x = input_tensor[0]
    y = input_tensor[1]
    r1 = ((x - 0.5) ** 2) + ((y - 0.5) ** 2)
    r2 = ((x - 0.6) ** 2) + ((y - 0.1) ** 2)
    result_1 = 0.80 * torch.exp(-r1 / (0.3 ** 2))
    result_2 = 0.88 * torch.exp(-r2 / (0.03 ** 2))
    combined_result = result_1 + result_2
    return combined_result.unsqueeze(0)

# Custom Logger Class (Updated to use 'Generation')
class GenerationInfoLogger(Logger):
    """
    Custom logger to print best parameters and objective value at each generation.
    """
    def __init__(self, searcher, interval=1, after_first_step=False):
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)
    
    def _log(self, status: dict):
        generation = status.get('iter', 0)  # Access the 'iter' key for generation count
        best_solution = status.get("best", None)
        if best_solution is not None:
            best_value = best_solution.evals.item()
            best_parameters = best_solution.values.tolist()
            print(f"Generation {generation}: Best parameters: {best_parameters}, Best objective value: {best_value:.4f}")

# Run the EvoTorch optimization multiple times
def run_optimization(repeats, generations, std_dev, job_name, seed):
    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    best_results = []  # To store the best results of each repeat
    global_count = 0
    local_count = 0

    for repeat in range(repeats):
        print(f"\nStarting optimization repeat {repeat + 1} of {repeats}...")

        # Set up the EvoTorch optimization problem (no rng_seed here)
        problem = Problem(
            'max',  
            bimodal_eval_func,  
            solution_length=2,  
            bounds=([0, 0], [1, 1]),  
            dtype=torch.float32
        )

        # Set up the genetic algorithm (no rng_seed here)
        searcher = GeneticAlgorithm(
            problem,
            popsize=200,
            operators=[GaussianMutation(problem, stdev=std_dev)]
        )

        # Initialize Loggers
        stdout_logger = StdOutLogger(searcher, interval=1)  # Logs every generation
        pandas_logger = PandasLogger(searcher)  # Logs data for analysis
        generation_info_logger = GenerationInfoLogger(searcher, interval=1)  # Custom logger

        # Run the optimization for the specified number of generations
        searcher.run(generations)

        # Debugging statements to inspect the best solution
        print("\nFinal Searcher Status:")
        print(f"Searcher status keys: {searcher.status.keys()}")
        print(f"Best solution object: {searcher.status['best']}")

        # Get the best solution after optimization
        best_solution = searcher.status["best"]
        best_value = best_solution.evals.item()
        best_parameters = best_solution.values.tolist()

        # Store the result of the current repeat
        best_results.append((best_parameters, best_value))

        # Print the best parameters and value for the current repeat
        print(f"\nBest parameters for repeat {repeat + 1}: {best_parameters}")
        print(f"Best objective value for repeat {repeat + 1}: {best_value:.4f}")

        # Log the result to a log file after each repeat
        with open("evotorch_best_results.log", "a") as log_file:
            log_file.write(f"Repeat {repeat + 1}: Best parameters: {best_parameters}, Best objective value: {best_value:.4f}\n")

        # Check if the best score is greater than 0.81
        if best_value > 0.81:
            global_count += 1
        else:
            local_count += 1

    # After all repeats, print and save the summary
    summary_filename = f"{job_name}_summary.txt"
    with open(summary_filename, "w") as summary_file:
        summary_file.write("\nSummary of Best Results for Each Repeat:\n")
        for i, (best_parameters, best_value) in enumerate(best_results):
            summary_file.write(f"Repeat {i + 1}: Best parameters: {best_parameters}, Best objective value: {best_value:.4f}\n")

        # Print Global and Local summary
        summary_file.write("\nGlobal and Local Summary:\n")
        summary_file.write(f"Global count (best score > 0.81): {global_count}\n")
        summary_file.write(f"Local count (best score <= 0.81): {local_count}\n")

    print(f"\nOptimization complete. Summary saved to {summary_filename}.")

if __name__ == "__main__":
    # Use argparse to accept user inputs from the command line
    parser = argparse.ArgumentParser(description="Run EvoTorch optimization multiple times.")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations per optimization.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the optimization.")
    parser.add_argument("--std_dev", type=float, default=0.2, help="Standard deviation for Gaussian mutation.")
    parser.add_argument("--job_name", type=str, default="evotorch_optimization_GA", help="Name of the SLURM job for the summary file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Run the optimization with the user-defined parameters
    run_optimization(args.repeats, args.generations, args.std_dev, args.job_name, args.seed)
