from skopt.space import Real, Integer


random_seed = 1
n_workers = 2
test_split_size = 0.2
cv_splits = 5
search_params = ['learning_rate', 'depth', 'l2_leaf_reg']
space = [
    Real(10**-3, 1, 'log-uniform', name=search_params[0]),
    Integer(1, 10, name=search_params[1]),
    Integer(1, 1000, name=search_params[2]),
]
n_calls = 20