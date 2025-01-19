import torch
import numpy as np
import time
from evotorch import Problem
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import GaussianMutation, OnePointCrossOver
import math
import pickle
import argparse
from datetime import datetime
import random  # Added to set Python's random seed

# ==========================
# 1. Data Generation
# ==========================

def gramacy_lee():
    """
    Generates x and y values for the Gramacy-Lee function.
    Handles the special case when x = 0 by assigning y = 5*pi + 1.

    Returns:
        x_vals (np.ndarray): Array of x values from -0.5 to 2.5 with step 0.001.
        y_vals (np.ndarray): Corresponding y values computed based on the Gramacy-Lee function.
    """
    x_vals = np.arange(-0.5, 2.501, 0.001)
    # Using np.where to handle x=0 efficiently
    y_vals = np.where(
        x_vals != 0,
        (np.sin(10 * np.pi * x_vals) / (2 * x_vals)) + ((x_vals - 1) ** 4),
        5 * math.pi + 1  # Correct handling for x = 0
    ).astype(np.float32)  # Ensuring float32 for consistency with PyTorch
    return x_vals, y_vals

# Precompute data once to avoid redundancy
x_vals, target_y_vals = gramacy_lee()

# ==========================
# 2. Trigonometric Interpolation
# ==========================

def trig_interpolation(x_vals, params):
    """
    Calculates the trigonometric interpolation based on input parameters.
    Vectorized for efficiency using NumPy operations.

    Args:
        x_vals (np.ndarray): Array of x values.
        params (list or np.ndarray): List of parameters [x0, x1, x2, ..., x64].

    Returns:
        np.ndarray: Predicted y values based on the interpolation.
    """
    y_vals = np.full_like(x_vals, params[0], dtype=np.float32)  # Initialize with x0
    n_max = (len(params) - 1) // 2  # Number of cosine and sine pairs

    for n in range(1, n_max + 1):
        a = 2 * n - 1  # Index for cosine coefficient
        b = 2 * n      # Index for sine coefficient
        y_vals += params[a] * np.cos(n * x_vals) + params[b] * np.sin(n * x_vals)

    return y_vals

# ==========================
# 3. Mean Squared Error Calculation
# ==========================

def mse_func(target, predicted):
    """
    Calculates the Mean Squared Error between target and predicted values.

    Args:
        target (np.ndarray): Actual y values.
        predicted (np.ndarray): Predicted y values.

    Returns:
        float: The computed MSE.
    """
    return np.mean((target - predicted) ** 2)

# ==========================
# 4. Evaluation Function for EvoTorch
# ==========================

def eval_trig_polynomial(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the MSE for a given set of parameters.

    Args:
        input_tensor (torch.Tensor): Tensor of parameters [x0, x1, x2, ..., x64].

    Returns:
        torch.Tensor: Tensor containing the MSE value.
    """
    params = input_tensor.cpu().numpy().tolist()  # Convert tensor to list
    predicted_y_vals = trig_interpolation(x_vals, params)  # Vectorized prediction
    error = mse_func(target_y_vals, predicted_y_vals)  # Compute MSE
    return torch.tensor(error, dtype=torch.float32)

# ==========================
# 5. Custom Logger Class
# ==========================

class CustomLogger:
    """
    Custom logger to log messages to a file and optionally to the console.
    """
    def __init__(self, filename=None):
        """
        Initializes the logger.

        Args:
            filename (str, optional): Filename for logging. Defaults to timestamped log file.
        """
        self.filename = filename or f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
    def log(self, message, print_to_console=True):
        """
        Logs a message with a timestamp.

        Args:
            message (str): The message to log.
            print_to_console (bool, optional): Whether to print the message to the console. Defaults to True.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        if print_to_console:
            print(formatted_message)
            
        with open(self.filename, 'a') as f:
            f.write(formatted_message + '\n')

# ==========================
# 6. Optimization Function
# ==========================

def run_optimization(repeats, generations, popsize, seed=None):
    """
    Runs the optimization process multiple times using a Genetic Algorithm.

    Args:
        repeats (int): Number of times to repeat the optimization.
        generations (int): Number of generations per optimization run.
        popsize (int): Population size for the Genetic Algorithm.
        seed (int, optional): Seed for reproducibility. Defaults to None.
    """
    logger = CustomLogger()
    best_results = []

    # Log initial configuration
    logger.log("\n" + "="*80)
    logger.log(f"Starting Optimization with:")
    logger.log(f"Number of Repeats: {repeats}")
    logger.log(f"Generations per Repeat: {generations}")
    logger.log(f"Population Size: {popsize}")
    if seed is not None:
        logger.log(f"Random Seed: {seed}")
    logger.log("="*80 + "\n")

    # If seed is provided, generate per-repeat seeds
    if seed is not None:
        rng = np.random.default_rng(seed)
        repeat_seeds = rng.integers(0, 2**32 - 1, size=repeats)
    else:
        repeat_seeds = [None] * repeats  # No seed; repeats are different each run

    for repeat in range(repeats):
        start_time = time.time()
        
        # Log repeat header
        logger.log("\n" + "="*40 + f" REPEAT {repeat + 1}/{repeats} " + "="*40)
        
        # Set seeds for reproducibility
        if repeat_seeds[repeat] is not None:
            seed_val = repeat_seeds[repeat]
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            random.seed(seed_val)
            # If using CUDA:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)
        
        solution_length = 65  # Number of parameters [x0, x1, ..., x64]
        
        # Define the optimization problem
        problem = Problem(
            'min',  # Minimization problem
            eval_trig_polynomial,  # Evaluation function
            solution_length=solution_length,
            bounds=([-1.0] * solution_length, [1.0] * solution_length),  # Parameter bounds
            dtype=torch.float32,
        )

        # Initialize the Genetic Algorithm
        searcher = GeneticAlgorithm(
            problem,
            popsize=popsize,
            operators=[
                OnePointCrossOver(problem, tournament_size=2),
                GaussianMutation(problem, stdev=0.2)
            ]
        )

        # Track best solution for this repeat
        best_solution_this_repeat = float('inf')
        best_params_this_repeat = None  # Initialize to store best parameters

        # Iterate over generations
        for gen in range(generations):
            searcher.step()  # Perform one generation
            
            # Retrieve current best evaluation
            current_best = searcher.status["best"].evaluation.item()
            current_best_params = searcher.status["best"].values.cpu().numpy().tolist()
            
            # Update best MSE and parameters if current best is better
            if current_best < best_solution_this_repeat:
                best_solution_this_repeat = current_best
                best_params_this_repeat = current_best_params
            
            # Log generation details with parameters
            logger.log(f"  Generation {gen + 1}/{generations}:")
            logger.log(f"    Current Best MSE: {current_best:.6f}")
            logger.log(f"    Best MSE So Far: {best_solution_this_repeat:.6f}")
            logger.log(f"    Population Size: {popsize}")
            if best_params_this_repeat is not None:
                # Format parameters to 6 decimal places for readability
                formatted_params = ", ".join([f"{param:.6f}" for param in best_params_this_repeat])
                logger.log(f"    Parameters for Best MSE So Far: [{formatted_params}]")
            else:
                logger.log(f"    Parameters for Best MSE So Far: None")
        
        # Retrieve final results for this repeat
        end_time = time.time()
        duration = end_time - start_time
        
        best_solution = searcher.status["best"]
        best_value = best_solution.evaluation.item()
        best_parameters = best_solution.values.cpu().numpy().tolist()
        
        best_results.append((best_parameters, best_value, duration))
        
        # Log repeat summary
        logger.log("\n" + "-"*30 + f" Repeat {repeat + 1} Summary " + "-"*30)
        logger.log(f"Final Best MSE: {best_value:.6f}")
        logger.log(f"Time Taken: {duration:.2f} seconds")
        logger.log("-"*80 + "\n")

    # Final Summary of All Repeats
    logger.log("\n" + "="*30 + " FINAL RESULTS " + "="*30)
    
    # Extract MSE values and runtimes
    mse_values = [result[1] for result in best_results]
    runtimes = [result[2] for result in best_results]

    best_mse = min(mse_values)
    worst_mse = max(mse_values)
    avg_mse = sum(mse_values) / len(mse_values)
    avg_runtime = sum(runtimes) / len(runtimes)
    
    logger.log(f"Best MSE Across All Repeats: {best_mse:.6f}")
    logger.log(f"Worst MSE Across All Repeats: {worst_mse:.6f}")
    logger.log(f"Average MSE Across All Repeats: {avg_mse:.6f}")
    logger.log(f"Average Runtime per Repeat: {avg_runtime:.2f} seconds")
    logger.log(f"Total Runtime: {sum(runtimes):.2f} seconds")
    
    # Identify the best result across all repeats
    best_overall = min(best_results, key=lambda x: x[1])
    best_overall_params = best_overall[0]
    best_overall_mse = best_overall[1]
    
    # Format best overall parameters
    formatted_best_overall_params = ", ".join([f"{param:.6f}" for param in best_overall_params])
    
    logger.log(f"Best Parameters Across All Repeats: [{formatted_best_overall_params}]")
    logger.log("="*80 + "\n")

    # Save results to a pickle file
    with open("EvoTorch_Interp.pkl", "wb") as f:
        pickle.dump(best_results, f)
    
    logger.log("Results saved to EvoTorch_Interp.pkl")

# ==========================
# 7. Main Execution
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvoTorch optimization multiple times.")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations per optimization.")
    parser.add_argument("--repeats", type=int, default=100, help="Number of times to repeat the optimization.")
    parser.add_argument("--popsize", type=int, default=5, help="Population size for the genetic algorithm.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Set global seeds if a seed is provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    run_optimization(args.repeats, args.generations, args.popsize, seed=args.seed)
