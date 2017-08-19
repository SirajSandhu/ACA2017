# optimization of movie_features and user_parameters : minimization of cost

#from scipy import optimize

def minimize_cost(movie_features, user_parameters, ratings_matrix, R) :

    import numpy as np

    #reshaping into row vectors
    #(num_movies, num_features) = movie_features.shape
    #(num_users, num_features) = user_parameters.shape

    #optimisation_vars = np.hstack((np.reshape(movie_features, (1, num_movies*num_features)),
    #                                                                            np.reshape(user_parameters, (1, num_users*num_features))))

    #optimisation_vars_grad = np.hstack((np.reshape(movie_features_grad, (1, num_movies*num_features)),
    #                                                                        np.reshape(user_parameters_grad, (1, num_users*num_features))))


    #import cost_function
    import gradients

    movie_features_grad = np.zeros(movie_features.shape)
    user_parameters_grad = np.zeros(user_parameters.shape)

    num_iters = 100
    learning_rate = 0.3
    reg_constant = 1

    for i in range(num_iters) :
        (movie_features_grad, user_parameters_grad) = gradients.compute_grad(movie_features, user_parameters, ratings_matrix, R, reg_constant)
        movie_features = movie_features - learning_rate*movie_features_grad
        user_parameters = user_parameters - learning_rate*user_parameters_grad

    return (movie_features, user_parameters)
