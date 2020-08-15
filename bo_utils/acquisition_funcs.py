import numpy as np
from scipy.stats import norm
from abc import ABC
import logging

logger = logging.getLogger(__name__)


# TODO: Determine interface for multi-variate evaluation, compatibility with predict
class AcquisitionFunctionBaseClass(ABC):
    """
    Abstract Base Class for specifying the interface of a generic Acquisition Function to be used in a BO loop.
    """

    def __init__(self, *args, **kwargs):
        """Base Initializer"""
        logger.debug("AcquisitionFunctionBaseClass object initialized.")

    def calculate(self, *args, **kwargs):
        """The all important API function call that will be exposed for calculating the acquisition function value."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.calculate(*args, **kwargs)


class PI(AcquisitionFunctionBaseClass):
    """Probability of Improvement acquisition function."""

    def __init__(self, epsilon: float, for_minimization: bool = False):
        """
        :param epsilon: float
            Exploration-exploitation trade-off parameter.
        :param for_minimization: bool
            Flag to indicate if the returned value is to be used for minimization or maximization. Essentially, when
            for_minimization is True, the calculated value is negated before returning.
        """
        super(PI, self).__init__()
        self.epsilon = epsilon
        self.for_minimization = for_minimization
        logger.debug("PI Acquisition Function Initialized.")

    def calculate(self, x, model, eta):
        """
        :param x: point to determine the acquisition value
        :param model: GP to predict target function value
        :param eta: best so far seen value
        :return: PI
        """
        x = np.array([x]).reshape([-1, 1])
        mu, sigma = model.predict(x, return_std=True)
        Z = (mu - eta - self.epsilon) / sigma
        pi = norm.cdf(Z)
        # return pi
        return -pi if self.for_minimization else pi


class EI(AcquisitionFunctionBaseClass):
    """Expected Improvement acquitision function."""

    def __init__(self, for_minimization: bool = False):
        """
        :param for_minimization: bool
            Flag to indicate if the returned value is to be used for minimization or maximization. Essentially, when
            for_minimization is True, the calculated value is negated before returning.
        """
        super(EI, self).__init__()
        self.for_minimization = for_minimization
        logger.debug("EI Acquisition Function Initialized.")

    def calculate(self, x, model, eta):
        """
        :param x: point to determine the acquisition value
        :param model: GP to predict target function value
        :param eta: best so far seen value
        :return: EI
        """
        x = np.array([x]).reshape([-1, 1])
        mu, sigma = model.predict(x, return_std=True)

        with np.errstate(divide='warn'):
            improvement = mu - eta
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        # return ei
        return -ei if self.for_minimization else ei


class ConfidenceBound(AcquisitionFunctionBaseClass):
    """Lower/Upper Confidence Bound Acquisition function (LCB/UCB)"""

    def __init__(self, kappa: float = 1.0, upper: bool = False):
        """
        :param kappa: float
            Multiplier for standard deviation. Default is 1.0.
        :param upper: bool
            When False (default), use LCB, otherwise use UCB.
        """
        super(ConfidenceBound, self).__init__()
        self.kappa = kappa
        self.upper = upper
        logger.debug("%s Acquisition Function Initialized." % ("UCB" if self.upper else "LCB"))

    def calculate(self, x, model):
        """
        Lower/Upper Confidence Bound, returns a value for the minimizer (or maximizer, respectively)
        :param x: point to determine the acquisition value
        :param model: GP to predict target function value
        :return: LCB/UCB
        """
        x = np.array([x]).reshape([-1, 1])
        mu, sigma = model.predict(x, return_std=True)

        value = mu - self.kappa * sigma * (-1 if self.upper else 1)
        return value
