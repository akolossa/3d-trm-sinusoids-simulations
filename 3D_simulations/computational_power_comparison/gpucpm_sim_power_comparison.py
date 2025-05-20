import gpucpm
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from parameters import parameters
import time
import csv

# Constants for 128μm³ scale
DIMENSION = 128  # 128 voxels = 128μm (1μm/voxel)
N_CELLS = 8      # Scaled from original 40,693 cells in 1cm³ to 8 cells in 128μm³

for simulation_n in range(1, 2):  # Single simulation run
    # Setup directories
    out_dir = '/home/arawa/liver-3d-simulations/3D_simulations/computational_power_comparison'
    #os.makedirs(out_dir, exist_ok=True)
    
    def run_sim():
        # Load liver microstructure (128μm³ scale)
        liver = np.load(f"/media/datadrive/arawa/Shabaz_simulation_22_04_25/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{DIMENSION}_new_adjusted.npz")["image"] #loads segmented image
        frc = (liver == 0).astype(np.uint32)  # Liver=0, BG=1
        
        # Set borders to BG
        frc[0] = frc[-1] = frc[:,0] = frc[:,-1] = frc[:,:,0] = frc[:,:,-1] = 1
        frc = frc * (1 + 2**24 * 2)  # Transform for CPM
        
        # Initialize simulation
        sim = gpucpm.Cpm(DIMENSION, 3, N_CELLS+1, parameters['temperature'], False)
        state = sim.get_state()
        
        # Add fixed background cell at center
        sim.add_cell(2, DIMENSION//2, DIMENSION//2, DIMENSION//2) 
        state[:,:,:] = frc
        
        # Add T-cells at random positions
        for i in range(N_CELLS):
            while True:
                x, y, z = [random.randint(0, DIMENSION-1) for _ in range(3)]
                if state[z,y,x] == 0:
                    sim.add_cell(1, x, y, z)
                    break

        # Set constraints
        sim.set_constraints(cell_type=1, target_area=parameters['target_area'], 
                          lambda_area=parameters['lambda_area'])
        sim.set_constraints(cell_type=1, lambda_perimeter=parameters['lambda_perimeter'],
                          target_perimeter=parameters['target_perimeter'])   
        sim.set_constraints(cell_type=1, other_cell_type=1, 
                          adhesion=parameters['adhesion_tcell_tcell'])
        sim.set_constraints(cell_type=1, lambda_act=parameters['lambda_act'],
                          max_act=parameters['max_act'])
        sim.set_constraints(cell_type=0, other_cell_type=1,
                          adhesion=parameters['adhesion_tcell_sinusoid'])
        sim.set_constraints(cell_type=2, fixed=1)
        sim.set_constraints(cell_type=2, other_cell_type=1,
                          adhesion=parameters['adhesion_tcell_bg']+2)
        sim.push_to_gpu()

        # Burn-in period
        sim.run(cell_sync=0, block_sync=0, global_sync=1,
               threads_per_block=4, positions_per_thread=32,
               positions_per_checkerboard=4, updates_per_checkerboard_switch=1,
               updates_per_barrier=1, iterations=parameters['burnin'],
               inner_iterations=1, shared=0, partial_dispatch=1)
        sim.synchronize()

        # Initialize tracking
        state = sim.get_state()
        explored_volume = np.zeros((DIMENSION, DIMENSION, DIMENSION), dtype=np.uint32)
        total_available_space = np.sum(liver != 0)
        centroids_all = []
        
        # Create metrics file
        csv_filename = f"./coverage_metrics_CPM_dim{DIMENSION}_{N_CELLS}cells_r{simulation_n}.csv"
        with open(csv_filename, 'w') as f:
            f.write("Time(min),Coverage(%),ExploredVoxels,MSD\n")

        # Main simulation loop
        for i in range(parameters['n_timepoints']):
            start_time = time.time()
            sim.push_to_gpu()  
            sim.run(cell_sync=0, block_sync=0, global_sync=1,
                   threads_per_block=4, positions_per_thread=32,
                   positions_per_checkerboard=4, updates_per_checkerboard_switch=1,
                   updates_per_barrier=1, iterations=parameters['runtime'],
                   inner_iterations=1, shared=0, partial_dispatch=1)    
            sim.synchronize()

            # Get metrics
            sim.pull_from_gpu()
            state = sim.get_state()
            cellids = state % 2**24
            
            # Update explored volume
            explored_volume[cellids > 1] = 1
            explored_voxels = np.sum(explored_volume)
            coverage = (explored_voxels / total_available_space) * 100
            
            # Calculate MSD
            centroids = sim.get_centroids()
            if i > 0:
                msd = np.mean(np.sum((centroids[1:] - centroids_all[-1])**2, axis=1))
            else:
                msd = 0
            centroids_all.append(centroids)
            print(f"Simulation completed in {(time.time()-start_time)/60:.2f} minutes")
            
            # Save metrics
            with open(csv_filename, 'a') as f:
                f.write(f"{i*parameters['min_per_iteration']},{coverage},{explored_voxels},{msd}\n")
            

        # Save final centroids
        np.save(f'./centroids_CPM_dim{DIMENSION}_{N_CELLS}cells_r{simulation_n}.npy', 
               np.array(centroids_all))

    if __name__ == "__main__":
        run_sim()
