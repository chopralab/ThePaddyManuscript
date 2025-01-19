import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
import paddy
from paddy.Paddy_Parameter import PaddyParameter


def gramacy_lee():
    """Return Gramacy and Lee function coordinates, vectorized version."""
    x_range = np.arange(-0.5, 2.501, 0.001)
    y_values = np.sin(10 * np.pi * x_range) / (2 * x_range) + (x_range - 1)**4
    return x_range.tolist(), y_values.tolist()

def mse_func(target, mse_input):
    """Return error of interpolation using numpy operations."""
    return np.mean((np.array(target) - np.array(mse_input))**2)

def poly(x_list, seed):
    """Return coordinates of a polynomial using vectorized operations."""
    x_array = np.array(x_list)
    coefficients = seed[:, 0]
    powers = np.arange(len(seed))[:, None] * np.ones_like(x_array)
    x_powers = x_array**powers
    return np.sum(coefficients[:, None] * x_powers, axis=0).tolist()

def trig_inter(x_list, seed):
    """Return coordinates of a trigonometric polynomial using vectorized operations."""
    if len(seed) % 2 == 0:
        raise ValueError("Must use odd value for dim greater than 1!")
    
    x_array = np.array(x_list)
    n_terms = (len(seed) - 1) // 2
    n_range = np.arange(1, n_terms + 1)
    
    cos_terms = np.cos(n_range[:, None] * x_array)
    sin_terms = np.sin(n_range[:, None] * x_array)
    
    a_coeffs = seed[1::2, 0]
    b_coeffs = seed[2::2, 0]
    
    result = (np.sum(a_coeffs[:, None] * cos_terms, axis=0) + 
             np.sum(b_coeffs[:, None] * sin_terms, axis=0) + 
             seed[0, 0])
    
    return result.tolist()

class EvalNumeric:
    def __init__(self, error_func=mse_func, t_func=gramacy_lee, f_func=trig_inter):
        self.error_func = error_func
        self.f_func = f_func
        self.x_vals, self.answer = t_func()
        self.iteration = 0  # Counter for iterations

    def eval(self, seed):
        self.iteration += 1  # Increment the counter
        y_val = self.f_func(self.x_vals, seed)
        mse = self.error_func(self.answer, y_val)
       # print(f"Count {self.iteration}: MSE for current seed: {mse}")
        return -mse

class Polynomial:
    def __init__(self, length, scope, gausian_type, normalization=True, limits=True):
        limit_init = [-scope, scope] if limits else None
        self.__dict__.update({
            f'polly{i}': PaddyParameter(
                param_range=[-scope, scope, scope*.05],
                param_type='continuous',
                limits=limit_init,
                gaussian=gausian_type,
                normalization=normalization
            ) for i in range(length)
        })

def run_optimization(n_runs=100):
    run_func = EvalNumeric()
    test_polly_space = Polynomial(length=65, scope=1, gausian_type='default', normalization=False)
    
    all_results = []
    for run_idx in range(n_runs):
        print(f"Starting run {run_idx + 1}...")
        test_runner = paddy.PFARunner(
            space=test_polly_space,
            eval_func=run_func.eval,
            rand_seed_number=25,
            yt=25,
            paddy_type='generational',
            Qmax=25,
            r=.02,
            iterations=9
        )
        
        start = time.time()
        print(f'Start time: {start}')
        test_runner.run_paddy(verbose='all')
        elapsed = time.time() - start
        
        
        fitness_list = test_runner.seed_fitness
        best_fitness = max(fitness_list)
        worst_fitness = min(fitness_list)
        avg_fitness = np.mean(fitness_list)
        best_params = test_runner.seed_params[np.argmax(fitness_list)]
        
        # print(f"Run {run_idx + 1} summary:")
        # print(f"  Best MSE: {-best_fitness:.6f}")
        # print(f"  Worst MSE: {-worst_fitness:.6f}")
        # print(f"  Average MSE: {-avg_fitness:.6f}")
        
        all_results.append({
            "best_params": best_params,
            "best_mse": -best_fitness,
            "worst_mse": -worst_fitness,
            "avg_mse": -avg_fitness,
            "time_elapsed": elapsed
        })
    
    print("\nOverall Summary:")
    for idx, result in enumerate(all_results, start=1):
        print(f"Run {idx}: Best MSE: {result['best_mse']:.6f}, ")
            #   f"Worst MSE: {result['worst_mse']:.6f}, "
            #   f"Avg MSE: {result['avg_mse']:.6f}, Time: {result['time_elapsed']:.2f}s")
        #print elapsed time
        print(f'Elapsed time: {result["time_elapsed"]:.2f} seconds')
    
    return all_results


if __name__ == "__main__":
    results = run_optimization(n_runs=100)
    # print("Optimization results:", results)
