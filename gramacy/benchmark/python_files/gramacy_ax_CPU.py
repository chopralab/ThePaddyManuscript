import math
import numpy as np
import random  # Imported to seed Python's random module
import time
import torch
from ax import optimize
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
import datetime
import os
import uuid
import traceback
import argparse
from tqdm import tqdm  # Imported for progress bars
import concurrent.futures  # Imported for parallel execution
import statistics  # Imported for calculating standard deviation


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

    # Spawn child seeds for each iteration
    child_seeds = seed_sequence.spawn(num_iterations)

    print(f"Child seeds: {child_seeds}")
    # Generate integer seeds from child SeedSequences
    return [int(child_seeds[i].generate_state(1)[0]) for i in range(num_iterations)]


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


def trig_inter_hp(params, device=torch.device("cpu")):
    """
    Calculates the trigonometric interpolation based on input parameters using PyTorch.
    Utilizes CPU for computations.

    Args:
        params (dict): Dictionary of parameters {'x0': value, 'x1': value, ..., 'x64': value}.
        device (torch.device): Device to perform computations on (CPU).

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
        ave_error = torch.mean(error)  # Scalar

        # Print out the Mean Squared Error
        print(f'Mean Squared Error: {ave_error:.4f}')
        return ave_error.item()
    except KeyError as e:
        raise ValueError(f"Missing parameter: {e}")
    except Exception as e:
        print(f"Error in trig_inter_hp: {e}")
        traceback.print_exc()
        raise


def run_iteration(iter_num, seed, device, total_runs, total_trials, sobol_num_trials, gpei_num_trials):
    """
    Runs a single optimization iteration using the Ax optimization library.

    Parameters:
        iter_num (int): The current iteration number (0-based index).
        seed (int): The seed for this iteration.
        device (torch.device): The device to run computations on.
        total_runs (int): Total number of runs (for logging purposes).
        total_trials (int): Total number of trials for optimization.
        sobol_num_trials (int): Number of trials for the SOBOL step.
        gpei_num_trials (int): Number of trials for the GPEI step.

    Returns:
        List: Contains best parameters, best value, and runtime.
    """
    try:
        start = time.time()
        run_num = iter_num + 1
        print(f"\n=== Starting Iteration {run_num}/{total_runs} ===")
        print(f'Total Trials: {total_trials}')

        # Set random seeds for reproducibility
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_torch_deterministic(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        print(f"Seed: {seed} (Type: {type(seed)})")

        # Define generation strategy using command-line arguments
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=sobol_num_trials,
                    max_parallelism=1,  # Set to 1 to fully utilize single GPU/CPU
                    model_kwargs={"seed": seed}
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=gpei_num_trials,
                    max_parallelism=1,  # Set to 1 to fully utilize single GPU/CPU
                ),
            ]
        )

        # Define parameter space for optimization
        parameters = [
            {"name": f"x{i}", "type": "range", "bounds": [-1.0, 1.0]} for i in range(65)
        ]

        # Initialize a mutable trial counter using a list
        trial_counter = [0]  # Using a list to allow mutation within the nested function

        # Define a wrapped evaluation function that includes run and trial information
        def wrapped_evaluation_function(params):
            """
            Nested evaluation function to track and log trial numbers within a run.

            Args:
                params (dict): Dictionary of parameters.

            Returns:
                dict: Dictionary containing the objective value.
            """
            # Increment the trial counter
            trial_counter[0] += 1
            trial_num = trial_counter[0]
            # Evaluate the objective function
            mse = trig_inter_hp(params, device=device)
            # Log the MSE with trial and run information
            print(f"MSE {mse:.4f} for trial {trial_num} of run {run_num}/{total_runs}")
            return {'trig_inter_hp': mse}

        # Running the optimization
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=wrapped_evaluation_function,
            objective_name='trig_inter_hp',
            minimize=True,
            total_trials=total_trials,
            generation_strategy=generation_strategy,
        )

        # Extract the best value correctly
        best_value = values[0]["trig_inter_hp"]  # Access the MSE value

        end = time.time()
        runtime = end - start

        # Log iteration results
        print(f"=== Iteration {run_num}/{total_runs} Completed ===")
        print(f"Best MSE: {best_value:.4f}")
        print(f"Time Taken: {runtime:.2f} seconds")
        print(f"Best Parameters Found: {best_parameters}")

        return [best_parameters, best_value, runtime]
    except Exception as e:
        print(f"Exception in Iteration {run_num}/{total_runs}: {e}")
        traceback.print_exc()
        return None


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Run Gramacy-Lee optimization with Ax.")
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
        default=1,
        help='Number of optimization iterations to run.'
    )
    parser.add_argument(
        '--parallel_workers',
        type=int,
        default=4,  # Set default to 4 for parallel execution
        help='Number of parallel workers. Defaults to 4.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Base seed for reproducibility. If not set, a random seed will be used.'
    )
    # *** New Arguments Start ***
    parser.add_argument(
        '--total_trials',
        type=int,
        default=10,
        help='Total number of trials for optimization.'
    )
    parser.add_argument(
        '--sobol_num_trials',
        type=int,
        default=2,
        help='Number of trials for the SOBOL step.'
    )
    parser.add_argument(
        '--gpei_num_trials',
        type=int,
        default=8,
        help='Number of trials for the GPEI step.'
    )
    # *** New Arguments End ***

    args = parser.parse_args()

    # Existing arguments
    save_directory = args.save_directory
    job_name = args.job_name
    num_iterations = args.num_iterations
    parallel_workers = args.parallel_workers  # Set to 4 by default
    total_trials = args.total_trials
    sobol_num_trials = args.sobol_num_trials
    gpei_num_trials = args.gpei_num_trials

    # Validation: Ensure that sobol_num_trials + gpei_num_trials == total_trials
    if sobol_num_trials + gpei_num_trials != total_trials:
        raise ValueError(
            f"The sum of sobol_num_trials ({sobol_num_trials}) and gpei_num_trials ({gpei_num_trials}) "
            f"must equal total_trials ({total_trials})."
        )

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

    print(f"Total number of trials per run: {total_trials}")
    print(f"SOBOL trials: {sobol_num_trials}")
    print(f"GPEI trials: {gpei_num_trials}")

    # Initialize seeds for each iteration
    seeds = initialize_seeds(base_seed, num_iterations)
    print("Generated child seeds.")

    # **Force CPU Usage**
    device = torch.device("cpu")
    print("Using CPU for computations.")

    # Prepare arguments for each iteration, including run_num and total_runs
    # **Modified Here:** Pass total_trials, sobol_num_trials, and gpei_num_trials
    args_list = [
        (iter_num, seeds[iter_num], device, num_iterations, total_trials, sobol_num_trials, gpei_num_trials)
        for iter_num in range(num_iterations)
    ]
    print(f"Prepared arguments for {num_iterations} iterations.")

    # **Implement Parallel Execution with Specified Workers**
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        # Submit all iterations to the executor
        future_to_iter = {
            executor.submit(run_iteration, *args_tuple): args_tuple[0] + 1  # run_num starts at 1
            for args_tuple in args_list
        }
        # Use tqdm to display progress
        for future in tqdm(concurrent.futures.as_completed(future_to_iter), total=len(future_to_iter), desc="Running Iterations"):
            run_num = future_to_iter[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception in run {run_num}: {e}")
                traceback.print_exc()

    # Save the results
    print("\nSaving the results to file...")
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

    if mse_values and runtimes:
        best_mse = min(mse_values)
        worst_mse = max(mse_values)
        average_mse = sum(mse_values) / len(mse_values)
        std_mse = statistics.stdev(mse_values) if len(mse_values) > 1 else 0.0
        average_runtime = sum(runtimes) / len(runtimes)
        std_runtime = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0

        # Print summary
        print("\n=== Summary of Optimization Runs ===")
        print(f"Number of Successful Runs: {len(mse_values)}")
        print(f"Best MSE: {best_mse:.4f}")
        print(f"Worst MSE: {worst_mse:.4f}")
        print(f"Average MSE: {average_mse:.4f}")
        print(f"Standard Deviation of MSE: {std_mse:.4f}")
        print(f"Average Runtime per Iteration: {average_runtime:.2f} seconds")
        print(f"Standard Deviation of Runtime: {std_runtime:.2f} seconds")
    else:
        print("No successful optimization runs to summarize.")


if __name__ == '__main__':
    main()
