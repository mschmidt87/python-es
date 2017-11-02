import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../es')

from es import natural_es as nes

TOLERANCE_1D = 1e-9
TOLERANCE_2D = 1e-9
MAX_ITER = 3000
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return (x - x0) ** 2

        res = nes.optimize(f, np.array(mu), np.array(sigma), max_iter=MAX_ITER)

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return (x[0] - x0) ** 2 + (x[1] - y0) ** 2

        res = nes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]), max_iter=MAX_ITER)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED


def test_rosenbrock():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (a, b) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(0.5, 1.5, 10), np.random.uniform(90., 110., 10))):

        theo_min = [a, a ** 2]

        def f(x):
            return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

        res = nes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]),
                           learning_rate_mu=0.2, learning_rate_sigma=0.001,
                           max_iter=100000)

        assert(abs(res['mu'][0] - theo_min[0]) < 1e-2), SEED
        assert(abs(res['mu'][1] - theo_min[1]) < 1e-2), SEED
