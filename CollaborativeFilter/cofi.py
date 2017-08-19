# collaborative filtering for movie ratings implemented in python

import pandas as pd
import numpy as np

# loading data set
ratings_matrix = pd.read_csv("movie_ratings.csv")
ratings_matrix = ratings_matrix.drop(['Name'], 1)
(num_movies, num_users) = ratings_matrix.shape

#===============================================================================

# generate R matrix : binary True, False to indicate if user j has rated movie i
R = ~ratings_matrix.isnull()

#===============================================================================

# initialising X, Theta matrices
num_features = 3

movie_features = np.random.randn(num_movies, num_features)
user_parameters = np.random.randn(num_users, num_features)

#===============================================================================

# mean squared error cost
#reg_constant = 0

#import cost_function
#cost = cost_function.compute_cost(movie_features, user_parameters, ratings_matrix, R, reg_constant)

#===============================================================================

# gradients of cost function wrt features and parameters
#import gradients
#(movie_features_grad, user_parameters_grad) = gradients.compute_grad(movie_features, user_parameters, ratings_matrix, R, reg_constant)

#===============================================================================

# learning : cost minimization
import optimization
(opt_movie_features, opt_user_parameters) = optimization.minimize_cost(movie_features, user_parameters, ratings_matrix, R)

#===============================================================================
