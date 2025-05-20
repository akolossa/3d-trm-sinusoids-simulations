import gpucpm
import numpy as np
from scipy import ndimage
from sys import argv
from save_vti_example import save_vti
import sys
sys.path.append('/home/arawa/Shabaz_simulation/figure_7_ak')
from parameters_bo import parameters

runtime = parameters['runtime']
version = parameters['version'] + '_EXAMPLE'
print('Starting simulation:', version)
temperature = parameters['temperature']  
max_act = parameters['max_act'] 
lambda_act = parameters['lambda_act']
adhesion = parameters['adhesion_tcell_sinusoid']
lambda_p = parameters['lambda_perimeter']
lambda_a = parameters['lambda_area']
dimension = 768
sim = gpucpm.Cpm(dimension, 3, 10, int( temperature ), False)

state = sim.get_state()

sim.add_cell(1, 32, 32, 32,)
sim.add_cell(1, 35,35, 35)
#sim.add_cell(1, 200,100, 200)
sim.set_constraints(cell_type=1, lambda_area =int(lambda_a), target_area = 150, 
        target_perimeter = 1400,
        lambda_perimeter = lambda_p, #makes lambda p integer
        max_act = int(max_act  ), lambda_act = int( lambda_act ) 
        )
sim.set_constraints(cell_type = 0, other_cell_type = 1, adhesion = adhesion  )

state[ 30:35, 30:35, 30:35 ] = 16777217 ## cell ID 1 

previous_centroid = np.array([32,32,32])
sim.push_to_gpu()
sim.run(cell_sync=0,block_sync=0,global_sync=1,
        threads_per_block = 4,
        positions_per_thread = 16,
        positions_per_checkerboard = 8,
        updates_per_checkerboard_switch = 1,
        updates_per_barrier = 1,
        iterations=10*8,
        inner_iterations=1,shared=0,partial_dispatch=1)

broken = False
average_displacement = 0
n_displacement = 0

for i in range( 0, runtime):
    sim.push_to_gpu()
    sim.run(cell_sync=0,block_sync=0,global_sync=1, 
        threads_per_block = 4,
        positions_per_thread = 32,
        positions_per_checkerboard = 8, 
        updates_per_checkerboard_switch = 1,
        updates_per_barrier = 1,
        iterations=8,
        inner_iterations=1,shared=0,partial_dispatch=1)
    #make sure gpu kernel finished
    sim.synchronize()

    # this call only retreives cell center positions from GPU, especially efficient
    # when running a large simulation, because memory transfer of whole sim state
    # to system memory is avoided
    centroids = sim.get_centroids()
    print(centroids)
    print('iteration', i, 'centroids:',centroids)
    activity = sim.get_act_state()
    print('activity:',activity)
    print(activity.shape)
    if i == 0:
        np.save('./activity.npy', activity[np.newaxis])
    else:
        cur_act = np.load()
        new_act = np.concatenate((cur_act, activity[np.newaxis]), axis=0)
        np.save('./activity.npy', new_act)

    sim.pull_from_gpu()
    state=sim.get_state()
    cellids = state % 2**24 #2^24 bits
    types = state // 2**24 #2^24 bits
    labeled_array, num_features = ndimage.label( state > 2**24, structure=np.ones((3,3,3)) )

    if i % 1 == 0 :
        new_centroid = centroids[0]
        print('new centroid:',new_centroid)
        displacement = np.sum((new_centroid - previous_centroid)**2)
        print('displacement:',displacement)
        average_displacement += displacement
        n_displacement += 1
        previous_centroid = new_centroid
    #print( num_features )
    component_sizes = sorted(np.bincount(labeled_array.ravel())[1:])
    if( len( component_sizes ) > 1 and component_sizes[0] > 3 ):
        #print( "cell is broken!" )
        #print( component_sizes )
        broken = True
        break
    # Save VTI file for the current timestep
    save_vti(types, i, dimension, version)


print('----------------')
print('Simulation finished')
print('----------------')
print('Parameters used:')
print('temperature:', temperature)
print('max_act:', max_act)
print('lambda_act:', lambda_act)
print('adhesion:', adhesion)
print('lambda_p:', lambda_p)
print('----------------')
print('Cell moving from centre:', int(average_displacement / n_displacement))
print('does cell break?', broken)
print('iterations:', i)
