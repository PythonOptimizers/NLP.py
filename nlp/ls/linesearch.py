"""A General module that implements various linesearch schemes."""

from math import sqrt
import numpy as np

__docformat__ = 'restructuredtext'


class LineSearchFailure(Exception):
    """Exception raised when a linesearch fails."""
    pass


class LineSearch(object):
    """A generic linesearch class.

    Most methods of this class should be overridden by subclassing.
    """

    def __init__(self, linemodel, name="Generic Linesearch", **kwargs):
        """Initialize a linesearch method.

            :parameters:
                :linemodel: ``C1LineModel`` or ``C2LineModel`` instance

            :keywords:
                :step: initial step size (default: 1.0)
                :value: initial function value (computed if not supplied)
                :slope: initial slope (computed if not supplied)
                :name: linesearch procedure name.
        """
        self.__linemodel = linemodel
        self.__name = name
        self._value = kwargs.get("value", linemodel.obj(0))
        self._slope = kwargs.get("slope", linemodel.grad(0))
        self.check_slope(self.slope)

        self._trial_value = self._value
        self._step = max(kwargs.get("step", 1.0), 0)

        eps = np.finfo(np.double).eps
        self._stepmin = sqrt(eps) / 100
        if self._step <= self.stepmin:
            raise LineSearchFailure("initial linesearch step too small")

        self._trial_iterate = None
        return

    @property
    def linemodel(self):
        """Return underlying line model."""
        return self.__linemodel

    @property
    def name(self):
        """Return linesearch procedure name."""
        return self.__name

    @property
    def value(self):
        """Return initial merit function value."""
        return self._value

    @property
    def step(self):
        """Return current step size."""
        return self._step

    @property
    def stepmin(self):
        """Return minimum permitted step size."""
        return self._stepmin

    @property
    def trial_value(self):
        """Return current merit function value."""
        return self._trial_value

    @property
    def iterate(self):
        """Return current iterate."""
        return self._trial_iterate

    @property
    def slope(self):
        """Return initial merit function slope in search direction."""
        return self._slope

    def check_slope(self, slope):
        if slope >= 0.0:
            raise ValueError("Direction must be a descent direction")

    def __iter__(self):
        # This method makes LineSearch objects iterable.
        return self

    def __next__(self):
        return self.next()

    def next(self):
        raise NotImplementedError("Please subclass")

    def _test(self, *args, **kwargs):
        """Linesearch satisfaction test."""
        raise NotImplementedError("Please subclass")


class ArmijoLineSearch(LineSearch):
    """Armijo backtracking linesearch."""

    def __init__(self, *args, **kwargs):
        """Instantiate an Armijo backtracking linesearch.

        The search stops as soon as a step size t is found such that

            ϕ(t) <= ϕ(0) + t * ftol * ϕ'(0)

        where 0 < ftol < 1 and ϕ'(0) is the directional derivative of
        a merit function f in the descent direction d. true.

        :keywords:
            :ftol: Value of ftol (default: 1.0e-4)
            :factor: Amount by which to reduce the steplength
                     during the backtracking (default: 1.5).
        """
        super(ArmijoLineSearch, self).__init__(*args, **kwargs)
        sqeps = sqrt(np.finfo(np.double).eps)
        self.__ftol = max(min(kwargs.get("ftol", 1.0e-4), 1 - sqeps), sqeps)
        self.__factor = max(min(kwargs.get("factor", 1.5), 100), 1.001)
        return

    @property
    def ftol(self):
        return self.__ftol

    @property
    def factor(self):
        return self.__factor

    def _test(self):
        """Test Armijo condition."""
        return (self.trial_value <= self.value + self.step * self.ftol * self.slope)

    def next(self):
        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step)

        if self._test():
            raise StopIteration

        self._step /= self.factor
        if self.step < self.stepmin:
            raise LineSearchFailure("linesearch step too small")
