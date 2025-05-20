parameters = { 
    "version": 'v1.BO',
    "dimension": 128,  
    "n_timepoints": 100, 
    "burnin": 60, 
    "runtime": 20, 
    "height_array": 100, 
    "nr_of_cells": 1, 
    # Regulates size/shape of cells
    "target_area": 150,  
    "lambda_area": 9,  # Optimized lambda_a
    "lambda_perimeter": 14 / 100,  # Optimized lambda_p, converted
    "target_perimeter": 1400,  
    "temperature": 33,  # Optimized temperature
    # Activity/polarization
    "max_act": 79,  # Optimized max_act
    "lambda_act": 959,  # Optimized lambda_act
    # Adhesion 
    "adhesion_tcell_sinusoid": 79,  # Optimized adhesion
    "adhesion_tcell_tcell": 0, 
    "adhesion_tcell_bg": 0 
}

#Best parameters found: [33, 959, 79, 50, 14, 9]
