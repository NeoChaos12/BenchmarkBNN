import numpy as np
import logging
from typing import Dict, Union, Callable
from .acquisition_funcs import AcquisitionFunctionBaseClass, EI
import ConfigSpace as cs
from scipy.optimize import minimize
from ..benchmarks import BenchmarkBaseClass

logger = logging.getLogger(__name__)


class MainLoop(object):
    """
    The main BO loop. Should be initialized with a target model, the target's configuration space, an acquisition
    function object, and a benchmark.

    The model object should expose an interface consisting of the following function signatures:

    - fit(X, y) -> None
    Called in order to fit a model to the given dataset (X, y), where X is an array-like of input data features and
    y is an array-like of corresponding regression target values.

    - predict(X) -> mean, variance
    Called in order to have a trained model predict a mean and a variance for each given data point, where X is an
    array-like of data features.

    The acquisition function object should be an instance of a sub-class of
    BenchmarkBNN.bo_utils.acquisition_funcs.AcquisitionFunctionBaseClass.

    The benchmark should be a benchmark object from HPOlib2, which when called returns a dictionary containing
    the requisite benchmark process results. More specifically, it must support support a callable attribute
    'get_configuration_space' which returns a ConfigSpace.ConfigurationSpace object which shall be used to read the
    domain of the BO search space, as well as be callable itself, thus returning a dictionary with various result
    values.
    """
    # TODO: Implement HPOlib2 benchmarks support, modify docstring

    def __init__(self, target: object, acquisition: AcquisitionFunctionBaseClass,
                 benchmark: BenchmarkBaseClass):
        """

        :param target: object
            The target model to be evaluated and benchmarked.
        :param acquisition: AcquisitionFunctionBaseClass
            The acquisition function to be used.
        :param benchmark: BenchmarkBaseClass
            The objective benchmark to be used for evaluating the target.
        """

        self.target = target
        self.acquisition = acquisition
        self.benchmark = benchmark
        logger.debug("Initialized main BO loop.")

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        required_attributes = {"fit", "predict"}
        for attr in required_attributes:
            if not hasattr(new_target, attr):
                raise AttributeError("The given target model of type %s does not support '%s'" %
                                     (type(new_target), attr))

        self._target = new_target

    @property
    def acquisition(self):
        return self._acq

    @acquisition.setter
    def acquisition(self, new_acq: AcquisitionFunctionBaseClass):
        if not isinstance(new_acq, AcquisitionFunctionBaseClass):
            raise TypeError("The given acquisition function must be an object of a sub-class of %s" %
                            AcquisitionFunctionBaseClass.__name__)
        self._acq = new_acq

    def main_loop(self, n_iterations: int, burn_in: Union[int, float]):
        # For 'n' iterations, loop over:
        # Perform burn-in, if required
        # Sample a new configuration for the target model
        # Train the model on the available data using the sampled configuration
        # Minimize the acquisition function
        # Acquire next evaluation point
        # Evaluate the objective at the next evaluation point
        #
        # The 'data' here consists of samples from the objective's configuration space and their objective function values.
        #

        search_space: cs.ConfigurationSpace = self.benchmark.get_config_space()
        search_bounds = [(p.lower, p.upper) for p in search_space.get_hyperparameters()]

        if isinstance(burn_in, float):
            # if burn_in < 0.0 or burn_in > 1.0:
            #     raise ValueError("When specifying a fraction, burn_in must belong to the closed interval [0.0, 1.0], "
            #                      "was %f." % burn_in)
            from math import floor
            burn_in = floor(burn_in * n_iterations)

        if not isinstance(burn_in, int):
            raise TypeError("The number of burn in steps to be taken must be an integer. Received type %s" %
                            type(burn_in))

        logger.info("Running main BO loop for %d iterations with %d burn in steps, using the search space:\n%s" %
                    (n_iterations, burn_in, str(search_space)))

        if burn_in <= 0 or burn_in >= n_iterations:
            logger.warning("Number of burn in steps should ideally be in the open interval (0, n_iterations), was %d" %
                           burn_in)

        X = []
        y = []
        incumbent = None

        for idx in range(n_iterations):
            if idx < burn_in:
                # Generate an initial dataset
                X.append(search_space.sample_configuration().get_array())
                y.append(self.benchmark(X[-1])["function_value"])
            else:
                # Perform BO using the current dataset
                self.target.fit(X, y)
                candidate = minimize(self.acquisition, x0=incumbent[0], bounds=search_bounds)
                # TODO: Update to HPOlib2 interface, currently using placeholders.
                yval = self.benchmark(candidate)["function_value"]
                X.append(candidate)
                y.append(yval)
            # Our convention: Incumbent has to have already been evaluated (no "theoretically sound" incumbents)
            incumbent = (X[-1], y[-1]) if incumbent is None or y[-1] < incumbent[1] else incumbent

        logger.info("Finished BO loop. Found final incumbent: %s" % str(incumbent))
