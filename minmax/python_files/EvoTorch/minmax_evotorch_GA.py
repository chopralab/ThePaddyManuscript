import torch
import numpy as np
import random
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger, Logger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, GaussianMutation
import argparse
from tqdm import tqdm
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

# Custom Logger Class
class GenerationInfoLogger(Logger):
    def __init__(self, searcher, interval=1, after_first_step=False):
        super().__init__(searcher, interval=interval, after_first_step=after_first_step)

    def _log(self, status: dict):
        generation = status.get('iter', 0)
        best_solution = status.get("best", None)
        if best_solution is not None:
            best_value = best_solution.evals.item()
            best_parameters = best_solution.values.tolist()
            print(f"Generation {generation}: Best parameters: {best_parameters}, Best objective value: {best_value:.4f}")

def run_optimization(repeats, generations, std_dev, job_name, seed):
    # Set seeds for reproducibility if a seed is provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    best_results = []
    global_count = 0
    local_count = 0

    for repeat in range(repeats):
        print(f"Starting optimization repeat {repeat + 1} of {repeats}...")

        # Set up the EvoTorch optimization problem without rng_seed
        problem = Problem(
            'max',
            bimodal_eval_func,
            solution_length=2,
            bounds=([0, 0], [1, 1]),
            dtype=torch.float32
        )

        searcher = GeneticAlgorithm(
            problem,
            popsize=200,
            operators=[
                OnePointCrossOver(problem, tournament_size=4),
                GaussianMutation(problem, stdev=std_dev)
            ]
            # No rng_seed argument here
        )

        StdOutLogger(searcher, interval=1)
        PandasLogger(searcher)
        GenerationInfoLogger(searcher, interval=1)

        searcher.run(generations)

        print(f"Searcher status keys: {searcher.status.keys()}")
        print(f"Best solution object: {searcher.status['best']}")

        best_solution = searcher.status["best"]
        best_value = best_solution.evals.item()
        best_parameters = best_solution.values.tolist()

        best_results.append((best_parameters, best_value))

        print(f"Best parameters for repeat {repeat + 1}: {best_parameters}")
        print(f"Best objective value for repeat {repeat + 1}: {best_value}")

        with open("evotorch_best_results.log", "a") as log_file:
            log_file.write(f"Repeat {repeat + 1}: Best parameters: {best_parameters}, Best objective value: {best_value}\n")

        if best_value > 0.81:
            global_count += 1
        else:
            local_count += 1

    summary_filename = f"{job_name}_GA_summary.txt"
    with open(summary_filename, "w") as summary_file:
        summary_file.write("\nSummary of Best Results for Each Repeat:\n")
        for i, (best_parameters, best_value) in enumerate(best_results):
            summary_file.write(f"Repeat {i + 1}: Best parameters: {best_parameters}, Best objective value: {best_value}\n")

        summary_file.write("\nGlobal and Local Summary:\n")
        summary_file.write(f"Global count (best score > 0.8): {global_count}\n")
        summary_file.write(f"Local count (best score <= 0.8): {local_count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvoTorch optimization multiple times.")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations per optimization.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat the optimization.")
    parser.add_argument("--std_dev", type=float, default=0.2, help="Standard deviation for Gaussian mutation.")
    parser.add_argument("--job_name", type=str, default="evotorch_optimization_GA", help="Name of the SLURM job for the summary file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    run_optimization(args.repeats, args.generations, args.std_dev, args.job_name, args.seed)
