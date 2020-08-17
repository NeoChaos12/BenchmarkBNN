import numpy as np
import ConfigSpace as cs
from typing import Union, Optional, Dict, List, Callable
from .benchmark_bases import SyntheticBenchmark
import logging

logger = logging.getLogger(__name__)


# Hartmann3_2 function definitions code adapted from dragonfly: https://github.com/dragonfly/dragonfly
def __hartmann3_2(x):
  """ Hartmann function in 3D. """
  alpha = np.array([1.0, 1.2, 3.0, 3.2])
  return __hartmann3_2_alpha(x, alpha)


def __hartmann3_2_alpha(x, alpha):
  """ Hartmann function in 3D with alpha. """
  pt = np.array(x)
  A = np.array([[3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]], dtype=np.float64)
  P = 1e-4 * np.array([[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]], dtype=np.float64)
  log_sum_terms = (A * (P - pt)**2).sum(axis=1)
  return alpha.dot(np.exp(-log_sum_terms))


__parameters = [
    cs.UniformFloatHyperparameter("x0", lower=0, upper=1),
    cs.UniformFloatHyperparameter("x1", lower=0, upper=1),
    cs.UniformFloatHyperparameter("x2", lower=0, upper=1),
]

hartmann3_2 = SyntheticBenchmark(ofunc=__hartmann3_2, parameters=__parameters, name="Hartmann3_2")
