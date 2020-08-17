import numpy as np
import ConfigSpace as cs
from typing import Union, Optional, Dict, List, Callable
from .benchmark_bases import SyntheticBenchmark
import logging

logger = logging.getLogger(__name__)


# Borehole_6 function definitions code adapted from dragonfly: https://github.com/dragonfly/dragonfly
def __borehole_6(x):
  """ Computes the Bore Hole function. """
  return __borehole_6_z(x, [1.0, 1.0])


def __borehole_6_z(x, z):
  """ Computes the Bore Hole function at a given fidelity. """
  # pylint: disable=bad-whitespace
  rw = x[0]
  L  = x[1] * (1680 - 1120.0) + 1120
  Kw = x[2] * (12045 - 9855) + 9855
  Tu = x[3]
  Tl = x[4]
  Hu = x[5]/2.0 + 990.0
  Hl = x[6]/2.0 + 700.0
  r  = x[7]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw + 0.001) * np.exp(z[1] - 1)
  f2 = 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl))
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  return f2 * z[0] + (1-z[0]) * f1


__parameters = [
    cs.UniformFloatHyperparameter("rw", lower=0.05, upper=0.15),
    cs.UniformFloatHyperparameter("L", lower=0, upper=1),
    cs.UniformFloatHyperparameter("Kw", lower=0, upper=1),
    cs.UniformIntegerHyperparameter("Tu", lower=63070, upper=115600),
    cs.UniformFloatHyperparameter("Tl", lower=63.1, upper=116),
    cs.UniformIntegerHyperparameter("Hu", lower=0, upper=240),
    cs.UniformIntegerHyperparameter("Hl", lower=0, upper=240),
    cs.UniformFloatHyperparameter("r", lower=100, upper=50000),
]

borehole_6 = SyntheticBenchmark(ofunc=__borehole_6, parameters=__parameters, name="Borehole_6")
