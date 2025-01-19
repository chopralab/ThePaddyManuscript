import math
import numpy as np
import random  # Imported to seed Python's random module
import time
import torch
import datetime
import os
import uuid
import traceback
import argparse
from tqdm import tqdm  # Imported for progress bars

# Total number of trials for optimization
total_trial_number = 1500

def set_torch_deterministic(seed):
    """
    Set PyTorch to be deterministic for reproducibility.

    Parameters:
    - seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_seeds(base_seed, num_iterations):
    """
    Initialize a list of unique, reproducible seeds for each iteration.

    Parameters:
    - base_seed (int): The base seed for reproducibility.
    - num_iterations (int): Number of unique seeds to generate.

    Returns:
    - List[int]: A list of integer seeds.
    """
    # Seed NumPy and Python's random module
    np.random.seed(base_seed)
    random.seed(base_seed)
    
    # Create a SeedSequence for generating child seeds
    seed_sequence = np.random.SeedSequence(base_seed)
    
    print(f"Child seeds: {seed_sequence.spawn(num_iterations)}")
    # Generate integer seeds from child SeedSequences
    return [int(seed_sequence.spawn(num_iterations)[i].generate_state(1)[0]) for i in range(num_iterations)]

def gramacy_lee():
    """
    Generates x and y values for the Gramacy-Lee function.
    Properly handles the special case when x = 0 by assigning y = 5*pi + 1.
    """
    x_vals = np.arange(-0.5, 2.501, 0.001, dtype=np.float32)
    y_vals = np.where(
        x_vals != 0,
        (np.sin(10 * np.pi * x_vals) / (2 * x_vals)) + ((x_vals - 1) ** 4),
        5 * math.pi + 1
    ).astype(np.float32)
    return x_vals, y_vals

# Precompute data once to avoid redundancy in evaluations
x_vals, target_y_vals = gramacy_lee()

def trig_inter_hp(params, device=torch.device("cpu"), iter_num=None, trial_num=None):
    """
    Calculates the trigonometric interpolation based on input parameters using PyTorch.
    Utilizes GPU acceleration if available.

    Args:
        params (dict): Dictionary of parameters {'x0': value, 'x1': value, ..., 'x64': value}.
        device (torch.device): Device to perform computations on (CPU or GPU).
        iter_num (int, optional): Current iteration number.
        trial_num (int, optional): Current trial number.

    Returns:
        float: The computed Mean Squared Error (MSE).
    """
    if len(params) % 2 == 0:
        raise ValueError("Must use an odd number of parameters greater than 1!")

    try:
        # Convert parameters to a PyTorch tensor and move to the specified device
        param_values = [params[f'x{i}'] for i in range(len(params))]
        params_tensor = torch.tensor(param_values, device=device, dtype=torch.float32)

        # Initialize prediction with x0 and ensure y_pred has shape [1, len(x_vals)]
        y_pred = params_tensor[0].unsqueeze(0).repeat(1, len(x_vals))  # Shape: [1, 3001]

        n_max = (len(params) - 1) // 2  # Number of cosine and sine pairs
        n_values = torch.arange(1, n_max + 1, device=device, dtype=torch.float32).unsqueeze(1)  # Shape: [n_max, 1]

        # Extract cosine and sine coefficients
        cos_coeffs = params_tensor[1:2 * n_max:2].unsqueeze(1)  # Shape: [n_max, 1]
        sin_coeffs = params_tensor[2:2 * n_max + 1:2].unsqueeze(1)  # Shape: [n_max, 1]

        # Compute n * x for all n and x
        # Shape: [n_max, len(x_vals)]
        x_tensor = torch.tensor(x_vals, device=device, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 3001]
        n_x = n_values * x_tensor  # Broadcasting to shape: [n_max, 3001]

        # Compute cos(n * x) and sin(n * x)
        cos_nx = torch.cos(n_x)    # Shape: [n_max, 3001]
        sin_nx = torch.sin(n_x)    # Shape: [n_max, 3001]

        # Multiply by coefficients and sum over n
        # Shape: [1, 3001]
        cos_term = torch.matmul(cos_coeffs.T, cos_nx)  # Shape: [1, 3001]
        sin_term = torch.matmul(sin_coeffs.T, sin_nx)  # Shape: [1, 3001]

        # Sum all terms to get the predicted y values
        y_pred += cos_term + sin_term  # Shape: [1, 3001]
        y_pred = y_pred.squeeze(0)  # Shape: [3001]

        # Compute error using torch.mean
        yp_tensor = torch.tensor(target_y_vals, device=device, dtype=torch.float32)  # Shape: [3001]
        error = (yp_tensor - y_pred).pow(2)  # Shape: [3001]
        ave_error = torch.mean(error).item()  # Scalar

        # Print Iteration, Trial, and MSE on a single line
        if iter_num is not None and trial_num is not None:
            print(f"Iteration {iter_num + 1} Trial {trial_num + 1} MSE: {ave_error:.4f}")
        
        return ave_error
    except KeyError as e:
        raise ValueError(f"Missing parameter: {e}")
    except Exception as e:
        print(f"Error in trig_inter_hp: {e}")
        traceback.print_exc()
        raise

def evaluation_function_wrapper(params, device, iter_num=None, trial_num=None):
    """
    Wrapper function to evaluate the MSE for given parameters.

    Args:
        params (dict): Dictionary of parameters.
        device (torch.device): Device to perform computations on.
        iter_num (int, optional): Current iteration number.
        trial_num (int, optional): Current trial number.

    Returns:
        float: The computed Mean Squared Error (MSE).
    """
    mse = trig_inter_hp(params, device=device, iter_num=iter_num, trial_num=trial_num)
    return mse

def run_iteration(iter_num, seed, device):
    """
    Runs a single optimization iteration using Random Search.

    Parameters:
        iter_num (int): The current iteration number.
        seed (int): The seed for this iteration.
        device (torch.device): The device to run computations on.

    Returns:
        List: Contains best parameters, best value, and runtime.
    """
    try:
        start = time.time()
        print(f"\nStarting iteration {iter_num + 1}...")
        print(f'Total trials: {total_trial_number}')

        # Set random seeds for reproducibility
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_torch_deterministic(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        print(f"Seed type: {type(seed)}; Seed value: {seed}")

        # Define parameter space boundaries
        param_bounds = {f'x{i}': (-1.0, 1.0) for i in range(65)}

        best_mse = float('inf')
        best_params = None

        # Run random trials with tqdm for progress bar
        for trial in tqdm(range(total_trial_number), desc=f"Iteration {iter_num + 1}"):
            # Randomly sample parameters within bounds
            params = {f'x{i}': random.uniform(*param_bounds[f'x{i}']) for i in range(65)}
            
            # Evaluate the MSE
            mse = evaluation_function_wrapper(params, device=device, iter_num=iter_num, trial_num=trial)
            
            # Update best parameters if current MSE is lower
            if mse < best_mse:
                best_mse = mse
                best_params = params

        end = time.time()
        runtime = end - start

        print(f"Iteration {iter_num + 1} completed. Time taken: {runtime:.2f} seconds")
        print("Best parameters found:")
        for param_name, param_value in sorted(best_params.items(), key=lambda x: int(x[0][1:])):  # Sort numerically by parameter index
            print(f"  {param_name}: {param_value:.6f}")
        print(f"Objective function value for best parameters: {best_mse:.4f}")


        return [best_params, best_mse, runtime]
    except Exception as e:
        print(f"Exception in iteration {iter_num + 1}: {e}")
        traceback.print_exc()
        return None

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run Gramacy-Lee optimization with Random Search.")
    parser.add_argument(
        '--save_directory',
        type=str,
        default="./results/",
        help='Directory to save the results'
    )
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help='Job name or identifier (optional)'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=100,
        help='Number of optimization iterations to run.'
    )
    parser.add_argument(
        '--parallel_workers',
        type=int,
        default=None,
        help='Number of parallel workers. Defaults to number of CPU cores.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Base seed for reproducibility. If not set, a random seed will be used.'
    )
    args = parser.parse_args()
    
    # Existing arguments
    save_directory = args.save_directory
    job_name = args.job_name
    num_iterations = args.num_iterations
    parallel_workers = args.parallel_workers or os.cpu_count()
    
    # Handle the seed
    if args.seed is not None:
        base_seed = args.seed
        print(f"Using fixed base seed: {base_seed}")
    else:
        base_seed = random.randint(0, 2**32 - 1)  # Generate a random seed
        print(f"No seed provided. Generated random base seed: {base_seed}")

    # Generate a timestamp with date and time
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")  # Format: YYMMDD_HHMMSS
    unique_id = uuid.uuid4().hex[:6]  # 6-character unique identifier
    job_id = job_name if job_name else os.environ.get('SLURM_JOB_ID', 'local')  # Get SLURM job ID if available

    # Define the save directory
    os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

    # Create a unique filename
    filename = os.path.join(
        save_directory,
        f"Paddy_Interp_{timestamp}_job{job_id}_{unique_id}.npy"
    )
    print(f"Generated timestamp: {timestamp}")
    print(f"Output filename: {filename}")
    print(f"Current Working Directory: {os.getcwd()}")

    print(f"Total number of trials: {total_trial_number}")

    # Initialize seeds for each iteration
    seeds = initialize_seeds(base_seed, num_iterations)
    print("Generated child seeds.")

    # Determine the device to use (GPU if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU for computations.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU for computations.")

    # Prepare arguments for each iteration
    args_list = [
        (iter_num, seeds[iter_num], device)
        for iter_num in range(num_iterations)
    ]
    print(f"Prepared arguments for {num_iterations} iterations.")

    # Run iterations serially with tqdm for progress bar
    results = []
    for args_tuple in tqdm(args_list, desc="Running iterations"):
        result = run_iteration(*args_tuple)
        results.append(result)

    # Save the results
    print("Saving the results to file...")
    hp_minmax_results = np.array(results, dtype=object)
    try:
        np.save(filename, hp_minmax_results)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Exception while saving results: {e}")
        traceback.print_exc()

    # Extract MSE values and runtimes
    mse_values = [entry[1] for entry in results if entry is not None]
    runtimes = [entry[2] for entry in results if entry is not None]

    if mse_values:
        best_mse = min(mse_values)
        worst_mse = max(mse_values)
        average_mse = sum(mse_values) / len(mse_values)
        std_dev_mse = np.std(mse_values)  # Compute standard deviation
        average_runtime = sum(runtimes) / len(runtimes)

        # Print summary
        print("\nSummary of Optimization Runs:")
        print(f"Best MSE: {best_mse:.4f}")
        print(f"Worst MSE: {worst_mse:.4f}")
        print(f"Average MSE: {average_mse:.4f}")
        print(f"Standard Deviation of MSE: {std_dev_mse:.4f}")
        print(f"Average Runtime per Iteration: {average_runtime:.2f} seconds")

        # **New Addition: Print All 65 Parameters for the Best MSE**
        # Find the index of the best MSE
        best_indices = [i for i, mse in enumerate(mse_values) if mse == best_mse]
        # In case multiple iterations have the same best MSE, select the first one
        best_index = best_indices[0]
        best_params_overall = results[best_index][0]

        print("\nBest Parameters Overall:")
        for param in sorted(best_params_overall.keys(), key=lambda x: int(x[1:])):  # Sort parameters numerically
            print(f"{param}: {best_params_overall[param]:.6f}")
    else:
        print("No successful optimization runs to summarize.")

if __name__ == '__main__':
    main()
