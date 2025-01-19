import numpy as np
import time
import torch
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from ax import optimize
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

# Suppress warnings from ax library
logging.getLogger('ax').setLevel(logging.ERROR)

def f_2(params):
    x = params['x']
    y = params['y']
    r1 = ((x - 0.5) ** 2) + ((y - 0.5) ** 2)
    r2 = ((x - 0.6) ** 2) + ((y - 0.1) ** 2)
    # Use NumPy's exp function for better performance
    return (0.80 * np.exp(-r1 / (0.3 ** 2))) + (0.88 * np.exp(-r2 / (0.03 ** 2)))

def custom_optimize(seed):
    try:
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=100, max_parallelism=20),
                GenerationStep(model=Models.GPEI, num_trials=400, max_parallelism=20),
            ]
        )

        best_parameters, best_values, experiment, model = optimize(
            parameters=[
                {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            evaluation_function=f_2,
            objective_name="objective",
            minimize=False,
            total_trials=500,
            random_seed=seed,
            generation_strategy=generation_strategy,
        )
        
    except (ValueError, RuntimeError) as e:
        print(f"Optimization encountered an error: {e}")
        return {"x": 0.0, "y": 0.0}, {"objective": 0.0}
    
    if isinstance(best_values, tuple):
        best_values = best_values[0]
    
    return best_parameters, best_values

def run_single_optimization(i):
    import random

    # Generate a unique random seed for each process
    seed = random.randint(0, 2**32 - 1)
    start_time = time.time()
    best_params, best_val = custom_optimize(seed)
    elapsed_time = time.time() - start_time

    solution_type = 'Global' if best_val['objective'] > 0.81 else 'Local'
    is_global_solution = 1 if solution_type == 'Global' else 0

    # Prepare the result data
    result = [best_params, best_val['objective'], elapsed_time, solution_type]

    # Optionally print detailed information
    print(f"Run {i + 1}: Best Parameters: {best_params}, Objective: {best_val['objective']:.4f}, "
          f"Time Taken: {elapsed_time:.2f} seconds, Solution Type: {solution_type}")

    return is_global_solution, result

def main():
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used for computations.")
    else:
        print("CUDA is not available. CPU will be used for computations.")

    global_solution_count = 0
    results_list = []
    total_runs = 100

    # Limit the number of worker processes to reduce memory usage
    max_workers = min(4, total_runs)  # Adjust the number based on your system's capabilities

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_optimization, i) for i in range(total_runs)]
        for future in as_completed(futures):
            try:
                is_global_solution, result = future.result()
                results_list.append(result)
                global_solution_count += is_global_solution

                # Update progress
                print(f"Global solution count: {global_solution_count}/{total_runs} (Global Solutions Found)")
            except Exception as e:
                print(f"An error occurred in a child process: {e}")

    np.save("Ax_MinMax_Results_Improved", np.array(results_list, dtype=object))
    print("Results saved successfully.")

    print("\nFinal Summary:")
    print(f"Total Runs: {total_runs}")
    print(f"Number of Global Solutions Found: {global_solution_count}")
    print(f"Percentage of Global Solutions: {(global_solution_count / total_runs) * 100:.2f}%")

    print("\nDetailed Results:")
    for idx, res in enumerate(results_list, start=1):
        params, objective, elapsed_time, solution_type = res
        print(f"Run {idx}: Best Parameters: {params}, Objective: {objective:.4f}, "
              f"Time Taken: {elapsed_time:.2f} seconds, Solution Type: {solution_type}")

if __name__ == '__main__':
    main()
