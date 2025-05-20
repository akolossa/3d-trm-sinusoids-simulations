from skopt.space import Integer

# Define parameter space
param_space=[
    Integer(5, 40, name='temperature'),
    Integer(500, 3000, name='lambda_act'),
    Integer(5, 100, name='max_act'),
    Integer(5, 80, name='adhesion'),
    Integer(5, 25, name='lambda_p'),
    Integer(5, 50, name='lamda_a'),
    #Real(50, 300, name='target_a'), # should be the same of what shabaz has used
    #Real(400, 3000, name='target_p') #should be the same of what shabaz has used
    ]


