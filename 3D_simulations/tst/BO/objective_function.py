import numpy as np
from scipy import stats, ndimage
import gpucpm

# Function to run the simulation
def run_simulation(temperature, max_act, lambda_act, adhesion, lambda_p, lambda_a):
    print(f"Running simulation with params: T={temperature}, max_act={max_act}, "
          f"lambda_act={lambda_act}, adhesion={adhesion}, lambda_p={lambda_p}", f"lambda_a={lambda_a}")
    try:
        dimension = 768
        sim = gpucpm.Cpm(dimension, 3, 10, int(temperature), False)
        sim.add_cell(1, 32, 32, 32)

        sim.set_constraints(
            cell_type=1,
            lambda_area=int(lambda_a),
            target_area=150,
            target_perimeter=1400,
            lambda_perimeter=lambda_p / 100,
            max_act=int(max_act),
            lambda_act=int(lambda_act)
        )
        sim.set_constraints(cell_type=0, other_cell_type=1, adhesion=int(adhesion))

        sim.push_to_gpu()

        # Tracking variables
        displacements = []
        broken = False
        previous_centroid = np.array([32, 32, 32])

        # Run simulation loop
        for i in range(200):
            sim.run(
                cell_sync=0, block_sync=0, global_sync=1,
                threads_per_block=4,
                positions_per_thread=32,
                positions_per_checkerboard=8,
                updates_per_checkerboard_switch=1,
                updates_per_barrier=1,
                iterations=8,
                inner_iterations=1, shared=0, partial_dispatch=1
            )
            sim.synchronize()

            centroids = sim.get_centroids()
            state = sim.get_state()

            # Check for broken cells
            labeled_array, num_features = ndimage.label(state > 2**24, structure=np.ones((3, 3, 3)))
            component_sizes = sorted(np.bincount(labeled_array.ravel())[1:])

            if len(component_sizes) > 1 and component_sizes[0] > 3:
                broken = True
                break

            # Calculate displacements
            if i % 5 == 0:
                new_centroid = centroids[0]
                displacement = np.linalg.norm(new_centroid - previous_centroid)
                displacements.append(displacement)
                previous_centroid = new_centroid

        avg_displacement = np.mean(displacements) if displacements else 0
        return displacements, broken, avg_displacement
    except Exception as e:
        print(f"Simulation failed: {e}")
        raise


def evaluate_simulation(params):
    print(f"Evaluating simulation with params: {params}")
    temperature, max_act, lambda_act, adhesion, lambda_p, lambda_a = params

    try:
        num_simulations = 50  # Number of simulations to run with the same parameters
        success_count = 0  # Count how many simulations meet the desired criteria
        total_penalty = 0  # To accumulate penalties

        for _ in range(num_simulations):
            # Run the simulation
            displacements, cell_broken, avg_disp = run_simulation(temperature, max_act, lambda_act, adhesion, lambda_p, lambda_a)

            # Penalize if the cell is broken
            if cell_broken:
                total_penalty += 10  # High penalty for broken cells 
                continue  # Skip further checks if the cell is broken

            # Analyze displacement data
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)

            # Check if displacements are in the desired range (5-25)
            penalty = 0
            for displacement in displacements:
                if displacement < 5:
                    penalty += 8  # Mild penalty for displacement < 5
                elif displacement > 35:
                    penalty += 8  # Higher penalty for displacement > 25

            # Normality test
            if len(displacements) >= 8:  # Minimum required for normality test
                _, p_value = stats.normaltest(displacements)
            else:
                p_value = 1.0  # Assume normality for small samples

            # Check the success criteria: displacements within range and normal distribution
            if penalty == 0 and p_value > 0.05:  # No penalty if displacements are within range and normally distributed
                success_count += 1
            else:
                total_penalty += penalty  # Add penalty for displacement issues

        # Calculate the success percentage
        success_percentage = success_count / num_simulations

        # If 80% or more simulations meet the criteria, no penalty, else apply penalty
        if success_percentage >= 0.9:
            penalty = 0
        else:
            penalty = 10  # Apply penalty if fewer than 80% of the simulations are successful

        total_penalty += penalty  # Add the success percentage penalty
        return total_penalty

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return float('inf')  # Penalize simulation failure
