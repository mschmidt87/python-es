import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import sys
from functools import partial
from matplotlib.colors import LogNorm
import seaborn as sns
from matplotlib import gridspec
cp = sns.color_palette()
sys.path.append('../')
from es import separable_natural_es as snes
from es import mo_natural_es as mo_nes
from es import exponential_natural_es as xnes
np.random.seed(123)

"""
This script tests whether the multi-objective NES finds 
multiple solutions to a multi-dimensional fitness function.
"""

def test_2d_fitness():
    def fitness_add(x):
        x = np.array(x).reshape(-1, 2)
        return np.sin(0.75*np.sum(x, axis=1))**2


    def fitness_subtract(x):
        x = np.array(x).reshape(-1, 2)
        return -10.*(np.diff(x, axis=1)[:, 0])**2


    def fitness(x):
        return np.array([fitness_add(x),
                         fitness_subtract(x)]).T


    mu = np.array([1., 1.])
    cov = np.array([[0.2, 0.], [0., 0.2]])*10.

    num_iter = 100
    population_size = 10

    res = mo_nes.optimize(fitness, cov, max_iter=num_iter,
                          record_history=True, population_size=population_size,
                          rng=12354)
    assert(np.allclose(res['mu'], np.pi/3., atol=0.1))
