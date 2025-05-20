import gpucpm
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from scipy import stats, ndimage






# Function to run the simulation
def run_simulation(temperature, max_act, lambda_act, adhesionsinusoids, adhesionborders, lambda_p, lambda_a):
    print(f"Running simulation with params: T={temperature}, max_act={max_act}, "
          f"lambda_act={lambda_act}, 'adhesionsinusoids={adhesionsinusoids},"
          f"adhesionborders={adhesionborders}, lambda_p={lambda_p}, lambda_a={lambda_a}")
    try:
        dimension = 128
        n_cells = 1
        n_timepoints = 100
        liver = np.load(f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}.npz")["image"] #loads sinusoids from paper
        frc = (liver == 0).astype(np.uint32) #converts to array where liver is 0 and bg is 1
        #set borders to 1
        frc[0] = 1
        frc[-1] = 1
        frc[:,0] = 1
        frc[:,-1] = 1
        frc[:,:,0] = 1
        frc[:,:,-1] = 1

        frc = frc * (1 + 2**24 * 2) #transforms index on the grid into bits for cpm 

        sim = gpucpm.Cpm(dimension, 3, n_cells, temperature, False) #initializes cpm on gpucpm
        state = sim.get_state() #sets empty sim (cpm)
        # Add fixed cell in the center- its a bg cell
        sim.add_cell(2, dimension//2, dimension//2, dimension//2) 
        print("loaded frc")
        state[:,:,:] = frc #sets state to frc - here we actually set  sinusoids/bg
        previous_centroids = np.zeros((n_cells, 3))  # Initialize array for centroids
        print('a')
        # Add t-cells
        for i in range(n_cells):
            while True:
                x = random.randint(0,dimension-1) #randomly selects x,y,z coordinates for t-cells
                y = random.randint(0,dimension-1)
                z = random.randint(0,dimension-1)
                if state[z,y,x] == 0:
                    sim.add_cell(1, x, y, z) #adds t-cells to simulation
                    previous_centroids[i] = np.array([x, y, z])  # Initialize previous centroid
                    break
    
        #constraints for t-cells
        sim.set_constraints(cell_type=1, target_area=150, lambda_area=lambda_a) #t-cell costraints
        sim.set_constraints(cell_type=1, lambda_perimeter=lambda_p, target_perimeter=1400)#t-cell constraints - what's the difference with above?   
        sim.set_constraints(cell_type=1, other_cell_type=1, adhesion=0) #sets adhesion between t-cells
        sim.set_constraints(cell_type=1, lambda_act=lambda_act, max_act=max_act)#t-cell - why all separate?? 
        #constraints for background/borders
        sim.set_constraints(cell_type=0, other_cell_type=1, adhesion=adhesionsinusoids) #sets adhesion btw t-cells and sinusoids - is it acually bg?
        sim.set_constraints(cell_type=2, fixed=1) #sets bg cell as fixed
        sim.set_constraints(cell_type=2, other_cell_type=1, adhesion=adhesionborders) #sets adhesion between bg and t-cells
        sim.push_to_gpu()
   
        # Burn-in - stabilise system before the main simulation run- is this done to avoid that initial conditions influence main sim?
        sim.run(cell_sync=0, block_sync=0, global_sync=1, #not sure i understand all param here
                threads_per_block=4, #allows for parallel processing of 4 threads per block (i.e. 4 cells) - threads are lightweight processes that execute independently
                positions_per_thread=16, #each thread processes 32 positions - so a total of 128 positions per block and 
                positions_per_checkerboard=8, #CHANGED FROM 4 TO 16
                updates_per_checkerboard_switch=1,
                updates_per_barrier=1,
                iterations=60,
                inner_iterations=1, shared=0, partial_dispatch=1)
        sim.synchronize()

        state = sim.get_state() #retrieves state of simulation 
        explored_volume = np.zeros((dimension, dimension, dimension), dtype=np.uint8) #####!!!!!!!!!!!
        avg_displacement = []
        tot_displacement = 0
        n_displacement = 0
        # Main simulation loop
        for i in range(0, n_timepoints):
            print("----------")
            print(f"Running timepoint {i + 1}/{n_timepoints}")
            sim.push_to_gpu()  
            # Run the simulation for the current timepoint
            sim.run(cell_sync=0, 
                    block_sync=0, 
                    global_sync=1, 
                    threads_per_block=4, 
                    positions_per_thread=16, 
                    positions_per_checkerboard=8, 
                    updates_per_checkerboard_switch=1, 
                    updates_per_barrier=1, 
                    iterations=20, 
                    inner_iterations=1, shared=0, 
                    partial_dispatch=1)    
            sim.synchronize()
            centroids = sim.get_centroids()
            sim.pull_from_gpu()  # Pull updated state from the GPU
            state = sim.get_state() #get current state of sim 
            cellids = state % 2**24
            types = state // 2**24
            print('1')
            explored_volume[cellids > 1] = 1 #keep track of explored volume
                        # Check for broken cells
            labeled_array, num_features = ndimage.label(state > 2**24, structure=np.ones((3, 3, 3)))
            component_sizes = sorted(np.bincount(labeled_array.ravel())[1:])
            print('2')
            if len(component_sizes) > 1 and any(size > 3 for size in component_sizes[1:]):
                print('Cell broken')
                broken = True
                break
            print('3')
            # Calculate displacements
            if i % 5 == 0:
                    # Track centroids
                new_centroids = centroids[1:]  # Remove the first cell which initializes sinusoids
                displacements = np.sum((new_centroids - previous_centroids)**2, axis=1)
                tot_displacement += displacements
                n_displacement += 1
                previous_centroids = new_centroids
            
                avg_displacement.append(np.mean(tot_displacement / n_displacement))  # CORRECT
                print(f"avg_disp: {avg_displacement}") 
        return broken, avg_displacement
    
    except Exception as e:
        print(f"Simulation failed: {e}")
        raise
            

def evaluate_simulation(params):
    print(f"Evaluating simulation with params: {params}")
    temperature, max_act, lambda_act, adhesionsinusoids, adhesionborders, lambda_p, lambda_a = params
    
    try:
        num_simulations = 10  # Number of simulations to run with the same parameters
        success_count = 0  # Count how many simulations meet the desired criteria
        total_penalty = 0  # To accumulate penalties

        for _ in range(num_simulations):
            # Run the simulation
            cell_broken, avg_disp = run_simulation(temperature, max_act, lambda_act, adhesionsinusoids, adhesionborders, lambda_p, lambda_a)

            # Penalize if the cell is broken
            if cell_broken:
                total_penalty += 10  # High penalty for broken cells 
                continue  # Skip further checks if the cell is broken

            # Analyze displacement data
            mean_disp = np.mean(avg_disp)

            # Check if displacements are in the desired range (5-25)
            penalty = 0
            for cell_displacements in avg_disp:
                    if cell_displacements < 5:
                        penalty += 10  # Mild penalty for displacement < 5
                    elif cell_displacements > 45:
                        penalty += 2  # Higher penalty for displacement > 25


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
        return 1e6  # Penalize simulation failure


