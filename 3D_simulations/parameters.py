
parameters = { 
    "dimension": 128,  
    "n_timepoints": 60000, 
    "version": f'FINAL',
    "burnin": 60, 
    "runtime": 20, 
    "height_array": 100, 
    #"nr_of_cells": 5, 
    # Regulates size/shape of cells
    "target_area": 150,  
    "lambda_area": 9,  # Optimized lambda_a
    "lambda_perimeter": 0.14,    # Optimized lambda_p
    "target_perimeter": 1400,  
    "temperature": 33,  # Optimized temperature
    # Activity/polarization
    "max_act": 79,  # Optimized max_act
    "lambda_act": 959,  # Optimized lambda_act
    # Adhesion 
    "adhesion_tcell_sinusoid": 50,  
    "adhesion_tcell_tcell": 120, 
    "adhesion_tcell_bg": 50, 
}
