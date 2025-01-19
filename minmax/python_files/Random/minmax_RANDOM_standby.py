import numpy as np
import time
import torch
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Optional: Suppress unnecessary logging (removed since Ax is no longer used)
# logging.getLogger('ax').setLevel(logging.ERROR)  # Removed

# Ensure deterministic behavior for reproducibility (optional)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Set device to CPU explicitly
device = torch.device("cpu")
print("Using CPU for computations.")

# Vectorized version of f_2 using PyTorch on CPU
def f_2_vectorized(params):
    """
    Vectorized implementation of f_2 using PyTorch on CPU.
    
    Parameters:
    - params: A dictionary containing tensors 'x' and 'y'.
    
    Returns:
    - result: Tensor containing the evaluated objective function.
    """
    x = params['x']
    y = params['y']
    r1 = ((x - 0.5) ** 2) + ((y - 0.5) ** 2)
    r2 = ((x - 0.6) ** 2) + ((y - 0.1) ** 2)
    result = (0.80 * torch.exp(-r1 / (0.3 ** 2))) + (0.88 * torch.exp(-r2 / (0.03 ** 2)))
    return result

def custom_random_optimize(seed, total_trials=500, batch_size=50):
    """
    Performs random optimization by sampling parameters uniformly at random.
    Evaluates in batches for efficiency.
    
    Parameters:
    - seed: Random seed for reproducibility.
    - total_trials: Total number of random samples to evaluate.
    - batch_size: Number of samples to evaluate in each batch.
    
    Returns:
    - best_params: Dictionary with the best 'x' and 'y' found.
    - best_objective: The highest objective value found.
    """
    try:
        # Seed the random number generators for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        best_params = {"x": 0.0, "y": 0.0}
        best_objective = -float('inf')  # Assuming we are maximizing

        for start in range(0, total_trials, batch_size):
            current_batch_size = min(batch_size, total_trials - start)
            # Sample random parameters uniformly from [0, 1]
            x_samples = torch.rand(current_batch_size)
            y_samples = torch.rand(current_batch_size)
            # Evaluate in a vectorized manner
            results = f_2_vectorized({"x": x_samples, "y": y_samples})
            # Convert results to numpy array
            results_cpu = results.numpy()
            # Find the best in the current batch
            batch_best_idx = np.argmax(results_cpu)
            batch_best_value = results_cpu[batch_best_idx]
            if batch_best_value > best_objective:
                best_objective = batch_best_value
                best_params = {
                    "x": x_samples[batch_best_idx].item(),
                    "y": y_samples[batch_best_idx].item()
                }

    except Exception as e:
        print(f"Random Optimization encountered an error: {e}")
        return {"x": 0.0, "y": 0.0}, 0.0

    return best_params, best_objective

def run_single_random_optimization(i, total_trials=500):
    """
    Runs a single Random optimization experiment.
    
    Parameters:
    - i: Index of the run (for logging purposes).
    - total_trials: Total number of random samples to evaluate.
    
    Returns:
    - is_global_solution: 1 if a global solution is found, else 0.
    - result: List containing best_params, best_val, elapsed_time, solution_type.
    """
    seed = random.randint(0, 2**32 - 1)
    start_time = time.time()
    best_params, best_val = custom_random_optimize(seed, total_trials)
    elapsed_time = time.time() - start_time

    solution_type = 'Global' if best_val > 0.81 else 'Local'
    is_global_solution = 1 if solution_type == 'Global' else 0

    # Prepare the result data
    result = [best_params, best_val, elapsed_time, solution_type]

    # Optionally print detailed information
    print(f"[Random] Run {i + 1}: Best Parameters: {best_params}, Objective: {best_val:.4f}, "
          f"Time Taken: {elapsed_time:.2f} seconds, Solution Type: {solution_type}")

    return is_global_solution, result

def main():
    """
    Main function to perform multiple random optimization runs in parallel using CPU.
    """
    global_solution_count_random = 0
    results_list_random = []
    total_runs = 100  # Number of parallel runs

    # Define the number of runs
    runs_per_optimizer = total_runs

    # Set the number of worker processes to 4
    max_workers = min(4, runs_per_optimizer)  # Ensure no more than 4 workers

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit Random optimization tasks
        futures_random = [executor.submit(run_single_random_optimization, i) for i in range(runs_per_optimizer)]
        
        for future in as_completed(futures_random):
            try:
                is_global_solution, result = future.result()
                results_list_random.append(result)
                global_solution_count_random += is_global_solution

                # Update progress
                print(f"[Random] Global solution count: {global_solution_count_random}/{total_runs} (Global Solutions Found)")
            except Exception as e:
                print(f"An error occurred in a Random child process: {e}")

    # Save results
    np.save("Random_MinMax_Results_Improved.npy", np.array(results_list_random, dtype=object))
    print("Random optimization results saved successfully.")

    # Print Final Summary
    print("\nFinal Summary for Random Optimizer:")
    print(f"Total Runs: {total_runs}")
    print(f"Number of Global Solutions Found: {global_solution_count_random}")
    print(f"Percentage of Global Solutions: {(global_solution_count_random / total_runs) * 100:.2f}%")

    # Detailed Results
    print("\nDetailed Results for Random Optimizer:")
    for idx, res in enumerate(results_list_random, start=1):
        params, objective, elapsed_time, solution_type = res
        print(f"Run {idx}: Best Parameters: {params}, Objective: {objective:.4f}, "
              f"Time Taken: {elapsed_time:.2f} seconds, Solution Type: {solution_type}")

if __name__ == '__main__':
    main()
