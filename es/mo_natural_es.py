import numpy as np
import pygmo

from . import lib


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def optimize(func, cov,
             learning_rates_sigma=None, learning_rate_A=None,
             population_size=None, max_iter=2000,
             fitness_shaping=True, record_history=False,
             rng=None):
    """
    Multi-objective natural evolution strategies.
    See Glasmachers, T., Schaul, T., & Schmidhuber, J. (2010). A
    natural evolution strategy for multi-objective optimization.
    Parallel Problem Solving from …, 8–11.
    """
    if not isinstance(cov, np.ndarray):
        raise TypeError('sigma needs to be of type np.ndarray')

    # Dimensionality of parameter space
    N = cov.shape[0]
    
    if learning_rates_sigma is None:
        (learning_rate_sigma_minus,
         learning_rate_sigma_plus) = lib.default_learning_rates_sigma_elitist(N)
    if learning_rate_A is None:
        learning_rate_A = lib.default_learning_rate_A_elitist(N)
    if population_size is None:
        population_size = lib.default_population_size_elitist(N)
    if not is_positive_definite(cov):
        raise ValueError('covariance matrix needs to be positive semidefinite')

    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(seed=rng)

    generation = 0
    history_mu = []
    history_cov = []
    history_pop = []
    history_fitness = []
    history_ranks = []
    
    # Find Cholesky decomposition of covariance matrix
    A = np.linalg.cholesky(cov).T
    assert(np.sum(np.abs(np.dot(A.T, A) - cov)) < 1e-12), 'Chochelsky decomposition failed'
    
    # Decompose A into scalar step and normalized covariance factor A
    sigma = (abs(np.linalg.det(A)))**(1. / N)
    A /= sigma

    if record_history:
        cov = sigma**2 * np.dot(A.T, A)
        # history_cov.append(cov.copy())
        # history_pop.append(np.empty((population_size, *np.shape(N))))

    mu = np.zeros((population_size, N))
    A = np.array([A for i in range(population_size)])
    sigma = np.array([[sigma] for i in range(population_size)])
    # Draw first population
    s = rng.normal(0, 1, size=(population_size, N))
    mu = mu + sigma * np.array([np.dot(A[i], s[i]) for i in range(population_size)])
    while True:
        # assert(abs(np.linalg.det(A) - 1.) < 1e-12), 'determinant of root '
        # 'of covariance matrix unequal one'

        s = rng.normal(0, 1, size=(population_size, N))
        mu_prime = mu + sigma * np.array([np.dot(A[i], s[i]) for i in range(population_size)])
        sigma_prime = sigma
        A_prime = A
        z = np.array([mu, mu_prime])
#        import pdb; pdb.set_trace()
        z_flattened = z.reshape(-1, np.shape(mu)[1])
        fitness = np.array([func(zi) for zi in z_flattened]).reshape(2, population_size, -1)
        fitness_flattened = fitness.reshape(-1, fitness.shape[-1])
        # Sort fitness values by non_dominant_sorting
        # We multiply by -1. because the pygmo routine is written for minimzation
        pareto_sets = pygmo.fast_non_dominated_sorting(-1. * fitness_flattened)[0]
        hv = pygmo.hypervolume(-1. * fitness_flattened)
        ref_point = hv.refpoint(offset=0.1)

        sorted_fitness = np.zeros((0, fitness_flattened.shape[1]))
        sorted_parameters = np.zeros((0, *(np.shape(z_flattened)[1:])))
        for pset in pareto_sets:
            pfront = fitness_flattened[pset]
            par = z_flattened[pset]
            hv_i = pygmo.hypervolume(-1. * pfront)
            delta_S = hv_i.contributions(ref_point)
            sorted_fitness = np.vstack((sorted_fitness,
                                        pfront[np.argsort(delta_S)[::-1]]))
            sorted_parameters = np.vstack((sorted_parameters,
                                           par[np.argsort(delta_S)[::-1]]))
        # Compute ranks for individuals
        ranks = np.array([np.where(sorted_parameters ==
                                   z_flattened[i])[0][0] for i in
                          range(z_flattened.shape[0])]).reshape(2, -1)
        # Update step sizes and covariance
        for i in range(population_size):
            if ranks[1][i] < ranks[0][i]:  # Offspring is more successful than parent
                sigma[i] *= np.exp(learning_rate_sigma_plus)
                sigma_prime[i] *= np.exp(learning_rate_sigma_plus)
                A_prime[i] *= (np.exp(learning_rate_A *
                                      (np.outer(s[i], s[i]) - np.eye(z.shape[-1]))))
            else:
                sigma[i] /= np.exp(learning_rate_sigma_minus)
                sigma_prime[i] /= np.exp(learning_rate_sigma_minus)

        # Copy N best individuals into the next generations
        mu = np.vstack((mu, mu_prime))[np.argsort(ranks.flatten())[:population_size]]
        sigma = np.vstack((sigma, sigma_prime))[np.argsort(ranks.flatten())[:population_size]]
        A = np.vstack((A, A_prime))[np.argsort(ranks.flatten())[:population_size]]
        
        if record_history:
            history_mu.append(mu.copy())
            cov = [sigma[i] ** 2 * np.dot(A[i].T, A[i]) for i in range(population_size)]
            history_cov.append(cov.copy())
            history_pop.append(z.copy())
            history_fitness.append(fitness.copy())
            history_ranks.append(ranks)
        generation += 1

        # exit if max iterations reached
        if generation > max_iter or np.min(sigma) ** 2 < 1e-20:
            break

    return {'mu': mu, 'sigma': sigma, 'history_mu': history_mu,
            'history_cov': history_cov, 'history_pop': history_pop,
            'history_fitness': history_fitness, 'history_ranks': history_ranks}
