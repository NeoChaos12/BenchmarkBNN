import ConfigSpace as cs
from typing import Union, Optional, Dict, List, Callable
from abc import ABC
import logging

logger = logging.getLogger(__name__)


class BenchmarkBaseClass(ABC):
    """Abstract Base Class for all Benchmarks used to specify the supported interface of any Benchmark objects."""

    def __init__(self):
        logger.debug("BenchmarkBaseClass initialized.")

    def get_configuration_space(self) -> cs.ConfigurationSpace:
        raise NotImplementedError("This method must be either implemented by sub-classing BenchmarkBaseClass or "
                                  "otherwise implemented by the object being used as a Benchmark.")

    def __call__(self) -> Dict:
        """ Upon being called, the benchmark must return a dictionary containing a key-value pair for the key
        'function_value'. """
        raise NotImplementedError("This method must be either implemented by sub-classing BenchmarkBaseClass or "
                                  "otherwise implemented by the object being used as a Benchmark.")


class SyntheticBenchmark(BenchmarkBaseClass):
    """A convenience class that makes it easy to specify synthetic benchmark objectives. Each Benchmark should be an
    instance of this class with the appropriate parameters."""

    def __init__(self, ofunc: Callable, parameters: List = None):
        self._ofunc = ofunc
        self.parameters = [] if parameters is None else parameters
        logger.debug("Synthetic Benchmark initialized.")

    def objective_function(self, *args, **kwargs):
        return self._ofunc(*args, **kwargs)

    def get_configuration_space(self, seed=None) -> cs.ConfigurationSpace:
        cspace = cs.ConfigurationSpace(name=self.__name__, seed=seed)
        cspace.add_hyperparameters(self.parameters)
        return cspace

    def __call__(self, *args, **kwargs):
        return {"function_value": self.objective_function(*args, **kwargs)}


placeholder_benchmark = SyntheticBenchmark(ofunc=lambda x: x, parameters=[cs.UniformFloatHyperparameter("x", lower=0.,
                                                                                                        upper=1.)])
