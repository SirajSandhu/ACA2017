# the mean squared error cost of the hypothesis, including L2 regularization

import numpy as np

def compute_cost(movie_features, user_parameters, ratings_matrix, R, reg_constant) :
    H = np.dot(movie_features, user_parameters.transpose()) - ratings_matrix.values
    H_rated = np.multiply(H, R)
    cost = np.nansum(np.square(H_rated)) + reg_constant*(np.square(movie_features).sum()) + reg_constant*(np.square(user_parameters).sum())
    return cost/2
