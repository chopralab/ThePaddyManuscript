import os
import math
import random
import time
import datetime
import uuid
import traceback
import argparse
import pickle

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def gramacy_lee():
    """
    Generates x and y values for the Gramacy-Lee function.
    Properly handles the special case when x = 0 by assigning y = 5*pi + 1 to prevent division by zero.
    
    Returns:
        xs (np.ndarray): Array of x values from -0.5 to 2.5 with step 0.001.
        y (np.ndarray): Corresponding y values computed based on the Gramacy-Lee function.
    """
    xs = np.round(np.arange(-0.5, 2.501, 0.001), 3).astype(np.float32)
    y = np.where(
        xs != 0,
        (np.sin(10 * np.pi * xs) / (2 * xs)) + ((xs - 1) ** 4),
        5 * np.pi + 1
    ).astype(np.float32)
    return xs, y

# Precompute x and y once
x, yp = gramacy_lee()

def trig_inter_hp(params, x=x, yp=yp):
    """
    Calculates the Mean Squared Error (MSE) between the predicted y-values
    and the actual y-values (Gramacy-Lee).
    
    Args:
        params (dict): Dictionary of parameters {'x0': value, 'x1': value, ..., 'x64': value}.
        x (np.ndarray): Array of x values.
        yp (np.ndarray): Array of actual y values.
    
    Returns:
        float: The computed MSE.
    """
    if len(params) % 2 == 0:
        print("Must use an odd number of parameters greater than 1!")
        return np.inf

    # Sort the params by key to enforce correct order
    try:
        params_sorted = [params[f'x{i}'] for i in range(len(params))]
    except KeyError as e:
        print(f"Missing parameter: {e}")
        return np.inf

    params_array = np.array(params_sorted, dtype=np.float32)
    x0 = params_array[0]

    cos_coeffs = params_array[1::2]  # x1, x3, ...
    sin_coeffs = params_array[2::2]  # x2, x4, ...

    n = np.arange(1, len(cos_coeffs) + 1, dtype=np.float32)
    n_x = np.outer(n, x)  # Shape: (Ncoeffs, len(x))

    cos_nx = np.cos(n_x)
    sin_nx = np.sin(n_x)

    y_pred = np.sum(
        cos_coeffs[:, np.newaxis] * cos_nx 
        + sin_coeffs[:, np.newaxis] * sin_nx, 
        axis=0
    ) + x0

    mse = np.mean((yp - y_pred) ** 2)
    return mse

def pspace_maker(leng):
    """
    Creates a Hyperopt parameter space with 'leng' parameters,
    each uniform(-1, 1).
    """
    return {f'x{i}': hp.uniform(f'x{i}', -1, 1) for i in range(leng)}

def run_single_optimization(iter_num, seed, space, max_evals):
    """
    Runs a single Hyperopt optimization iteration.
    
    Args:
        iter_num (int): The iteration number.
        seed (int): Random seed for reproducibility.
        space (dict): Hyperopt parameter space.
        max_evals (int): Number of evaluations for Hyperopt.
    
    Returns:
        tuple: (best_params, best_mse, runtime, sorted_params)
    """
    try:
        print(f"\n[Iteration {iter_num + 1}] Starting with Seed: {seed}")
        print(f"[Iteration {iter_num + 1}] Running {max_evals} trials.")

        trials = Trials()
        
        # We will store MSEs in a list instead of printing them as we go
        mse_values = []

        def objective(params):
            loss = trig_inter_hp(params)
            # Append MSE to the list (no printing here)
            mse_values.append(loss)
            return {"loss": loss, "status": STATUS_OK}

        start_time = time.time()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(seed)
        )
        end_time = time.time()
        runtime = end_time - start_time

        # Now that all trials are done, we can print them in order
        print(f"\n[Iteration {iter_num + 1}] Summary of all trials:")
        for i, val in enumerate(mse_values, start=1):
            print(f"    Trial {i}/{max_evals} => MSE: {val:.6f}")

        # Find best MSE from trials
        best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
        best_mse = trials.trials[best_trial_idx]['result']['loss']
        best_param_values = trials.trials[best_trial_idx]['misc']['vals']

        # Convert best_param_values into a sorted array
        param_dict = {k: v[0] for k, v in best_param_values.items()}
        sorted_params = [param_dict[f'x{i}'] for i in range(len(param_dict))]

        print(f"[Iteration {iter_num + 1}] Completed in {runtime:.2f} seconds.")
        print(f"[Iteration {iter_num + 1}] Best MSE: {best_mse:.6f}")

        return (best, best_mse, runtime, sorted_params)

    except Exception as e:
        print(f"[Iteration {iter_num + 1}] Exception: {e}")
        traceback.print_exc()
        return (None, None, None, None)


def initialize_seeds(base_seed, num_iterations):
    """
    Initializes a list of unique seeds for each iteration.
    If base_seed is None, generates a random one.
    """
    if base_seed is None:
        base_seed = int(time.time() * 1000) & ((1 << 32) - 1)
        print(f"Generated random base seed: {base_seed}")
    else:
        print(f"Using provided base seed: {base_seed}")
    
    np.random.seed(base_seed)
    random.seed(base_seed)
    seed_sequence = np.random.SeedSequence(base_seed)
    child_seeds = seed_sequence.spawn(num_iterations)
    return [int(child_seeds[i].generate_state(1)[0]) for i in range(num_iterations)], base_seed

def main():
    parser = argparse.ArgumentParser(description="Run Hyperopt optimization for Gramacy-Lee function.")
    parser.add_argument(
        '--save_directory',
        type=str,
        # Update the default to your desired path:
        default="/scratch/gilbreth/iyer95/pad/gramacy/pickle",
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
        default=10,
        help='Number of optimization iterations.'
    )
    parser.add_argument(
        '--max_evals',
        type=int,
        default=250,
        help='Number of evaluations per Hyperopt optimization.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Base seed for random number generation.'
    )
    args = parser.parse_args()

    save_directory = args.save_directory
    job_name = args.job_name
    num_iterations = args.num_iterations
    max_evals = args.max_evals

    # Generate a timestamp and unique ID
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    # If job_name wasn't provided, fall back to SLURM_JOB_ID or "local"
    job_id = job_name if job_name else os.environ.get('SLURM_JOB_ID', 'local')

    # Make sure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Initialize seeds
    seeds, base_seed = initialize_seeds(args.seed, num_iterations)

    # Construct filenames for saving
    filename_npy = os.path.join(
        save_directory,
        f"Hyperopt_Interp_{timestamp}_job{job_id}_seed{base_seed}_{unique_id}.npy"
    )
    filename_pkl = filename_npy.replace('.npy', '.pkl')

    print(f"Generated timestamp: {timestamp}")
    print(f"Output NPY filename: {filename_npy}")
    print(f"Output PKL filename: {filename_pkl}")
    print(f"Current Working Directory: {os.getcwd()}")

    # Define the parameter space (65 parameters)
    fspace = pspace_maker(65)

    # Store all iteration results
    results = []

    # Run each iteration in a simple for-loop
    for iter_num in range(num_iterations):
        print(f"\n===== Running iteration {iter_num + 1} of {num_iterations} =====")
        result = run_single_optimization(iter_num, seeds[iter_num], fspace, max_evals)
        results.append(result)

    # Bundle results with seed info
    results_with_seed = {
        'base_seed': base_seed,
        'iteration_seeds': seeds,
        'results': results
    }

    # Save both .npy and .pkl
    try:
        np.save(filename_npy, results_with_seed)
        with open(filename_pkl, 'wb') as f:
            pickle.dump(results_with_seed, f)

        print(f"\nResults saved successfully!")
        print(f"  - NPY file: {filename_npy}")
        print(f"  - PKL file: {filename_pkl}")
    except Exception as e:
        print(f"\nException while saving results: {e}")
        traceback.print_exc()

    # Evaluate valid results (filter out any None)
    valid_results = [
        (mse, runtime, params)
        for _, mse, runtime, params in results
        if None not in (mse, runtime, params)
    ]

    if valid_results:
        mse_values, runtimes, all_params = zip(*valid_results)
        best_idx = np.argmin(mse_values)
        best_mse = mse_values[best_idx]
        worst_mse = max(mse_values)
        average_mse = np.mean(mse_values)
        average_runtime = np.mean(runtimes)
        best_params = all_params[best_idx]

        print("\nSummary of Optimization Runs:")
        print(f"Base Seed Used: {base_seed}")
        print(f"Best MSE: {best_mse:.6f}")
        print(f"Worst MSE: {worst_mse:.6f}")
        print(f"Average MSE: {average_mse:.6f}")
        print(f"Average Runtime per Iteration: {average_runtime:.2f} seconds")

        print("\nBest Parameters Found:")
        for i, param in enumerate(best_params):
            print(f"x{i}: {param:.6f}")
    else:
        print("\nNo successful optimization runs to summarize.")

if __name__ == '__main__':
    main()
