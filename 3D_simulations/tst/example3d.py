import gpucpm
import numpy as np
from scipy import ndimage
from sys import argv

dimension = 64
sim = gpucpm.Cpm(dimension, 3, 10, int( argv[1] ), False)

state = sim.get_state()

sim.add_cell(1, 32, 32, 32,)
#sim.add_cell(1, 200,200, 100)
#sim.add_cell(1, 200,100, 200)
sim.set_constraints(cell_type=1, lambda_area = int(argv[6]), target_area = 150, 
        target_perimeter = 1400,
        lambda_perimeter = float( argv[5] )/100, #makes lambda p integer
        max_act = int( argv[3] ), lambda_act = int( argv[2] ) 
        )
sim.set_constraints(cell_type = 0, other_cell_type = 1, adhesion = int( argv[4] )  )

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

for i in range( 0, 50 ):
    sim.push_to_gpu()
    sim.run(cell_sync=0,block_sync=0,global_sync=1, 
        threads_per_block = 4,
        positions_per_thread = 16,
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
    print("Centroids shape:", centroids.shape)
    sim.pull_from_gpu()
    state=sim.get_state()
    labeled_array, num_features = ndimage.label( state > 2**24, structure=np.ones((3,3,3)) )

    if i % 5 == 0 :
        new_centroid = centroids[0]
        print('new centroid:', new_centroid)
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
print( argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], average_displacement / n_displacement, broken, i )
