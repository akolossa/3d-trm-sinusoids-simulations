# TODO: WHY the sinusoids are activated - not sure but managed to see activity of one cell only
# TODO: Analyse cell velocity with different n_cells simulations 
import gpucpm
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from vti_visualisation_closeSinusoids import save_vti
from parameters import parameters
import time
import csv
from calculate_metrics import track_metrics

for simulation_n in range(4,5):
    # set stuff up 
    dimension = 128
    half_dimension = dimension // 2
    height_array = parameters['height_array']
    width_array = height_array
    dimension_minus = dimension - height_array
    n_timepoints = parameters['n_timepoints'] 
    version = 'FinalVersion'
    n_cells = 2

    out_dir = f"/media/datadrive/arawa/Shabaz_simulation_22_04_25/figure_7_ak/current_vti_files/output_{version}_dim{dimension}_{n_cells}cells_r{simulation_n}"
    os.makedirs(out_dir, exist_ok=True)


    def run_sim():
        liver = np.load(f"/media/datadrive/arawa/Shabaz_simulation_22_04_25/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")["image"] #loads segmented image
        #liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")["image"] #loads segmented image
        #liver = np.load(f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}.npz")["image"] #loads sinusoids from paper
        frc = (liver == 0).astype(np.uint32) #converts to array where liver is 0 and bg is 1
        #set borders to 1
        frc[0] = 1
        frc[-1] = 1
        frc[:,0] = 1
        frc[:,-1] = 1
        frc[:,:,0] = 1
        frc[:,:,-1] = 1

        frc = frc * (1 + 2**24 * 2) #transforms index on the grid into bits for cpm 
        print('CELL NUMBER:',n_cells)

        sim = gpucpm.Cpm(dimension, 3, n_cells+1, parameters['temperature'], False) #initializes cpm on gpucpm
        state = sim.get_state() #sets empty sim (cpm)
        # Add fixed cell in the center- its a bg cell
        sim.add_cell(2, dimension//2, dimension//2, dimension//2) 
        print("loaded frc")
        state[:,:,:] = frc #sets state to frc - here we actually set  sinusoids/bg
        previous_centroids = np.zeros((n_cells, 3))  # Initialize array for centroids
        
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
        #print("previous centroids", previous_centroids)

        #constraints for t-cells
        sim.set_constraints(cell_type=1, target_area=parameters['target_area'], lambda_area=parameters['lambda_area']) #t-cell costraints
        sim.set_constraints(cell_type=1, lambda_perimeter=parameters['lambda_perimeter'], target_perimeter=parameters['target_perimeter'])#t-cell constraints - what's the difference with above?   
        sim.set_constraints(cell_type=1, other_cell_type=1, adhesion=parameters['adhesion_tcell_tcell']) #sets adhesion between t-cells
        sim.set_constraints(cell_type=1, lambda_act=parameters['lambda_act'], max_act=parameters['max_act'])#t-cell - why all separate?? 
        #constraints for background/borders
        sim.set_constraints(cell_type=0, other_cell_type=1, adhesion=parameters['adhesion_tcell_sinusoid']) #sets adhesion btw t-cells and sinusoids - is it acually bg?
        sim.set_constraints(cell_type=2, fixed=1) #sets bg cell as fixed
        sim.set_constraints(cell_type=2, other_cell_type=1, adhesion=parameters['adhesion_tcell_bg']+2) #sets adhesion between bg and t-cells
        sim.push_to_gpu()

        # Burn-in - stabilise system before the main simulation run- is this done to avoid that initial conditions influence main sim?
        sim.run(cell_sync=0, block_sync=0, global_sync=1, #not sure i understand all param here
                threads_per_block=4, 
                positions_per_thread=32, #each thread processes 32 positions - so a total of 128 positions per block and 
                positions_per_checkerboard=4, #CHANGED FROM 4 TO 16
                updates_per_checkerboard_switch=1,
                updates_per_barrier=1,
                iterations=parameters['burnin'],
                inner_iterations=1, shared=0, partial_dispatch=1)
        sim.synchronize()

        state = sim.get_state() #retrieves state of simulation 
        explored_volume = np.zeros((dimension, dimension, dimension), dtype=np.uint32) #####!!!!!!!!!!!

        # Prepare for metric collection
        centroids_all = []
        # all_metrics = []
        # tot_displacement = np.zeros(n_cells)
        # n_displacement = np.zeros(n_cells)
        # activity = sim.get_act_state()
        total_available_space = np.sum(liver != 0)  # Total available space in the liver
        # File path to save data
        csv_filename = f"/media/datadrive/arawa/Shabaz_simulation_22_04_25/figure_7_ak/exploration_data/exploration_data_{version}_dim{dimension}_{n_cells}cells_r{simulation_n}.csv"

        # Open CSV file in write mode ONCE (before the loop)
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timepoint", "Fraction_Explored", "Raw_Explored_Voxels"])  # Write header


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
                    positions_per_thread=32, 
                    positions_per_checkerboard=4, 
                    updates_per_checkerboard_switch=1, 
                    updates_per_barrier=1, 
                    iterations=parameters['runtime'], 
                    inner_iterations=1, shared=0, 
                    partial_dispatch=1)    
            sim.synchronize()

            # Get the centroids of the current timepoint
            centroids = sim.get_centroids()
            # print("Centroids shape:", centroids.shape)
            # print(centroids)

            start_time = time.time()
            start_time1 = time.time()
            sim.pull_from_gpu()  # Pull updated state from the GPU
            state = sim.get_state() #get current state of sim 
            cellids = state % 2**24
            # types = state // 2**24
     
            explored_volume[cellids > 1] = 1 #keep track of explored volume - ensures that only the voxels that have been explored are marked as 1
            #above means that if a cell revisits the same voxel, it will not increment the count of explored voxels because that voxel is already marked as explored
            explored_voxels = np.sum(explored_volume) #is it counting the voxels that have been explored? yes, the line above ensures that 
            # Compute fraction explored
            fraction_explored = explored_voxels / total_available_space
            # print(f'total available space: {total_available_space}')
            # print(f"Fraction of space explored at timepoint {i + 1}: {fraction_explored:.4f}")
            # print(f"Explored voxels at timepoint {i + 1}: {explored_voxels}")
            # activity = sim.get_act_state()
            #print('activity:',activity)
            # np.save(f'/home/arawa/Shabaz_simulation/figure_7_ak/activity', activity)
            # Append to CSV instead of overwriting
            end_time1 = time.time()
            print('time1:', (end_time1 - start_time1)/60)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, fraction_explored, explored_voxels])
            centroids = sim.get_centroids()
            centroids_all.append(centroids)
            end_time = time.time()
            print('time:', (end_time - start_time)/60)
            #save_vti(types, centroids, i, dimension, version, simulation_n, n_cells, exploration=explored_volume, max_distance=14, activity=activity)

        # Save all centroids to a .npy file
        np.save(f'/media/datadrive/arawa/Shabaz_simulation_22_04_25/figure_7_ak/centroids/centroids_{version}_dim{dimension}_{n_cells}cells_r{simulation_n}.npy', np.array(centroids_all))

            # previous_centroids, tot_displacement, n_displacement, metrics = track_metrics(sim, 
            #                                                                             state, 
            #                                                                             liver, 
            #                                                                             centroids, 
            #                                                                             previous_centroids, 
            #                                                                             tot_displacement, 
            #                                                                             n_displacement, 
            #                                                                             i, 
            #                                                                             metrics_csv_file=f"/home/arawa/Shabaz_simulation/figure_7_ak/current_vti_files/output_{version}_dim{dimension}_{n_cells}cells_r{simulation_n}/simulation_metrics_{version}_dim{dimension}_{n_cells}cells_r{simulation_n}.csv", 
            #                                                                             voxel_volume=1)
            # all_metrics.extend(metrics)
            #print("Metrics:", all_metrics)
            
    if __name__ == "__main__":
        start_time = time.time()
        out_dir = run_sim()
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print(f"run_sim() took {elapsed_time:.2f} minutes")