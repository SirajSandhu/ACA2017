# the gradients of the mean squared error cost function wrt elements of movie_features and user_parameters

import numpy as np

def compute_grad(movie_features, user_parameters, ratings_matrix, R, reg_constant) :
    H = np.dot(movie_features, user_parameters.transpose()) - ratings_matrix.values
    H_rated = (H*R).values

    (num_movies, num_features) = movie_features.shape
    (num_users, num_features) = user_parameters.shape

    movie_features_grad = np.zeros(movie_features.shape)
    user_parameters_grad = np.zeros(user_parameters.shape)

    for i in range(num_movies) :
        row_i = (np.array([H_rated[i]])).transpose()
        movie_features_grad[i] = np.nansum(user_parameters*row_i, 0)

    for i in range(num_users) :
        col_i = (np.array([H_rated[:,i]])).transpose()
        user_parameters_grad[i]=np.nansum(movie_features*col_i, 0)

    movie_features_grad = movie_features_grad + reg_constant*movie_features
    user_parameters_grad = user_parameters_grad + reg_constant*user_parameters

    return (movie_features_grad, user_parameters_grad)
