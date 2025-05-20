from skopt.space import Integer

# Define parameter space
param_space=[
    Integer(5, 40, name='temperature'),
    Integer(500, 3000, name='lambda_act'),
    Integer(5, 100, name='max_act'),
    Integer(5, 80, name='adhesionsinusoids'),
    Integer(0,30, name='adhesionborders'),
    Integer(5, 25, name='lambda_p'),
    Integer(5, 50, name='lamda_a'),
    ]


