from objective_function import evaluate_simulation
from parameter_space_setup import param_space
from skopt import gp_minimize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Function to store intermediate results
def store_intermediate_results(opt_result, iteration, filename='/home/arawa/tst/BO/bo_resultsFV.txt'):
    try:
        # Open file in append mode to store intermediate results
        with open(filename, 'a') as f:
            f.write(f"Iteration {iteration}: Best parameters: {opt_result.x}, Best score: {opt_result.fun}\n")
    except Exception as e:
        print(f"Error storing intermediate results: {e}")

# Run Bayesian Optimization
print("Starting Bayesian Optimization...")
try:
    # Create the callback function to store intermediate results
    def callback(opt_result):
        # Call the store_intermediate_results function with current iteration results
        iteration = len(opt_result.x_iters)  # Number of iterations completed
        store_intermediate_results(opt_result, iteration)
    
    # Run the optimization with the callback
    result = gp_minimize(
        evaluate_simulation,
        dimensions=param_space,
        n_calls=50,
        random_state=42,
        callback=[callback]  # Pass callback to the optimizer
    )
    
    print("Optimization complete.")
    print("Best parameters found:", result.x)
    
    # Save final results to a file
    with open('/home/arawa/tst/BO/bo_resultsFV.txt', 'w') as f:
        f.write("Optimization complete.\n")
        f.write("Best parameters found: " + str(result.x) + "\n")
        f.write("Best score: " + str(result.fun) + "\n")

except Exception as e:
    print(f"Optimization failed: {e}")
    with open('/home/arawa/tst/BO/bo_resultsFV.txt', 'w') as f:
        f.write(f"Optimization failed: {e}\n")
