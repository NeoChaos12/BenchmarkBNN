import ConfigSpace as cs
from typing import Union, Optional, Dict, List, Callable
from abc import ABC
import logging

logger = logging.getLogger(__name__)


class BenchmarkBaseClass(ABC):
    """ Abstract Base Class for all Benchmarks used to specify the supported interface of any Benchmark objects. """

    def __init__(self):
        logger.debug("BenchmarkBaseClass initialized.")

    def get_configuration_space(self) -> cs.ConfigurationSpace:
        raise NotImplementedError("This method must be either implemented by sub-classing BenchmarkBaseClass or "
                                  "otherwise implemented by the object being used as a Benchmark.")

    def __call__(self, *args, **kwargs) -> Dict:
        """ Upon being called, the benchmark must return a dictionary containing a key-value pair for the key
        'function_value'. """
        raise NotImplementedError("This method must be either implemented by sub-classing BenchmarkBaseClass or "
                                  "otherwise implemented by the object being used as a Benchmark.")


class SyntheticBenchmark(BenchmarkBaseClass):
    """ A convenience class that makes it easy to construct synthetic benchmark objectives. Each Benchmark should be an
    instance of this class with the appropriate parameters. """

    def __init__(self, ofunc: Callable, parameters: List[cs.hyperparameters.Hyperparameter] = None,
                 name: str = "SyntheticBenchmark"):
        """
        Initialize new synthetic benchmark with the given properties.
        :param ofunc: Callable
            A function that accepts an array of values corresponding to the configuration space as input and returns a
            single real number. Internally, this function performs the evaluation of all given inputs when the
            benchmark object itself is called.
        :param parameters: List
            A list of ConfigSpace.hyperparameters.Hyperparameter objects, corresponding to the configuration space of
            this benchmark.
        :param name: str
            The name of this benchmark.
        """
        super().__init__()
        self._ofunc = ofunc
        self.parameters = [] if parameters is None else parameters
        self.name = name
        logger.debug("Synthetic Benchmark initialized.")

    def __objective_function(self, *args, **kwargs):
        """ Wrapper to call the underlying objective function. """
        return self._ofunc(*args, **kwargs)

    def get_configuration_space(self, seed=None) -> cs.ConfigurationSpace:
        """ Reads the list of parameters for this benchmark and returns a corresponding ConfigSpace.ConfigurationSpace
        object. """
        cspace = cs.ConfigurationSpace(name=self.name, seed=seed)
        cspace.add_hyperparameters(self.parameters)
        logger.debug("Constructed Configuration Space %s:\n" % str(cspace))
        return cspace

    def __call__(self, *args, **kwargs):
        """ Forwards the given arguments to the underlying objective function and wraps the output in the required
        dictionary format. """
        return {"function_value": self.__objective_function(*args, **kwargs)}


# An ultra-simple synthetic benchmark that doesn't really do much, useful as a placeholder.
placeholder_benchmark = SyntheticBenchmark(
    ofunc=lambda x: x,
    parameters=[cs.UniformFloatHyperparameter("x", lower=0., upper=1.)],
    name="PlaceholderBenchmark",
)
