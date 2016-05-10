# -*- coding: utf-8 -*-
"""A general linesearch framework."""

from math import sqrt
import numpy as np
from nlp.tools.exceptions import LineSearchFailure

__docformat__ = 'restructuredtext'

eps = np.finfo(np.double).eps
sqeps = sqrt(eps)


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
            :trial_value: function value at `x + t*d`
                          (computed if not supplied)
            :name: linesearch procedure name.
        """
        self.__linemodel = linemodel
        self.__name = name

        self._value = kwargs.get("value", None)
        if self._value is None:
            self._value = linemodel.obj(0)

        self._slope = kwargs.get("slope", None)
        if self._slope is None:
            self._slope = linemodel.grad(0)
        self.check_slope(self.slope)

        self._step0 = max(kwargs.get("step", 1.0), 0)
        self._step = self._step0

        self._stepmin = sqrt(eps) / 100
        if self._step <= self.stepmin:
            raise LineSearchFailure("initial linesearch step too small")

        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = kwargs.get("trial_value", None)
        if self._trial_value is None:
            self._trial_value = self.linemodel.obj(self.step, x=self.iterate)
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
        """Return current full-space iterate.

        While not normally needed by the linesearch procedure, it is here so it
        can be recovered on exit and doesn't need to be recomputed.
        """
        return self._trial_iterate

    @property
    def slope(self):
        """Return initial merit function slope in search direction."""
        return self._slope

    def check_slope(self, slope):
        """Check is supplied direction is a descent direction."""
        if slope >= 0.0:
            msg = "Direction must be a descent direction: slope = %8.1e" % slope
            raise ValueError(msg)

    def __iter__(self):
        # This method makes LineSearch objects iterable.
        self._step = self._step0  # reinitialize search
        return self

    def __next__(self):
        return self.next()

    def next(self):
        raise NotImplementedError("Please subclass")


class ArmijoLineSearch(LineSearch):
    """Armijo backtracking linesearch."""

    def __init__(self, *args, **kwargs):
        u"""Instantiate an Armijo backtracking linesearch.

        The search stops as soon as a step size t is found such that

            ϕ(t) <= ϕ(0) + t * ftol * ϕ'(0)

        where 0 < ftol < 1 and ϕ'(0) is the directional derivative of
        a merit function f in the descent direction d. true.

        :keywords:
            :ftol: constant used in Armijo condition (default: 1.0e-4)
            :bkmax: maximum number of backtracking steps (default: 20)
            :decr: factor by which to reduce the steplength
                   during the backtracking (default: 1.5).
        """
        name = kwargs.pop("name", "Armijo linesearch")
        super(ArmijoLineSearch, self).__init__(*args, name=name, **kwargs)
        self.__ftol = max(min(kwargs.get("ftol", 1.0e-4), 1 - sqeps), sqeps)
        self.__bkmax = max(kwargs.get("bkmax", 20), 0)
        self.__decr = max(min(kwargs.get("decr", 1.5), 100), 1.001)
        self._bk = 0
        return

    @property
    def ftol(self):
        return self.__ftol

    @property
    def bkmax(self):
        return self.__bkmax

    @property
    def decr(self):
        return self.__decr

    @property
    def bk(self):
        return self._bk

    def next(self):
        if self.trial_value <= self.value + self.step * self.ftol * self.slope:
            raise StopIteration()

        self._bk += 1
        if self._bk > self.__bkmax:
            raise LineSearchFailure("backtracking limit exceeded")

        self._step /= self.decr
        if self.step < self.stepmin:
            raise LineSearchFailure("linesearch step too small")

        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step, x=self.iterate)

        return self.step


class ArmijoWolfeLineSearch(ArmijoLineSearch):
    """Improved Armijo backtracking linesearch."""

    def __init__(self, *args, **kwargs):
        u"""Instantiate an improved Armijo backtracking linesearch.

        The search stops as soon as a step size t is found such that

            ϕ(t) <= ϕ(0) + t * ftol * ϕ'(0)

        where 0 < ftol < 1 and ϕ'(0) is the directional derivative of
        a merit function f in the descent direction d. true.

        :keywords:
            :ftol: constant used in Armijo condition (default: 1.0e-4)
            :gtol: constant used in curvature condition (default: 0.9999)
            :bkmax: maximum number of Armijo backtracking steps (default: 20)
            :wmax: maximum number of expansion steps (default: 5)
            :incr: factor by which to increase the steplength during the
                   loose Wolfe search (default: 5.0)
            :decr: factor by which to reduce the steplength
                   during the backtracking (default: 1.5).
        """
        name = kwargs.pop("name", "Armijo-Wolfe linesearch")
        super(ArmijoWolfeLineSearch, self).__init__(*args, name=name, **kwargs)

        self.__gtol = min(max(kwargs.get("gtol", 0.9999),
                              self.ftol + sqeps),
                          1 - sqeps)
        self.__wmax = max(kwargs.get("wmax", 5), 0)
        self.__incr = max(min(kwargs.get("incr", 5.0), 100), 1.001)
        self._nw = 0

        self._trial_slope = self.linemodel.grad(self.step, x=self.iterate)
        return

    @property
    def gtol(self):
        return self.__gtol

    @property
    def wmax(self):
        return self.__wmax

    @property
    def incr(self):
        return self.__incr

    @property
    def trial_slope(self):
        return self._trial_slope

    def next(self):
        goal = self.value + self.step * self.ftol * self.slope
        armijo = self.trial_value <= goal

        # Increase step to try and satisfy loose Wolfe condition.
        wolfe = self.trial_slope >= self.gtol * self.slope
        if self._nw < self.wmax and armijo and not wolfe:
            step = self.step
            self._nw += 1
            self._step *= self.incr
            self._trial_iterate = self.linemodel.x + \
                self.step * self.linemodel.d
            self._trial_value = self.linemodel.obj(self.step)
            self._trial_slope = self.linemodel.grad(self.step)
            return step

        if self._bk > self.bkmax:
            raise LineSearchFailure("backtracking limit exceeded")

        if armijo:
            raise StopIteration()

        hz = self.trial_value <= self.value + 1.0e-10 * abs(self.value)
        if self.trial_slope <= -0.8 * self.slope and hz:
            raise StopIteration()

        self._bk += 1
        self._step /= self.decr
        if self.step < self.stepmin:
            raise LineSearchFailure("linesearch step too small")

        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step)
        self._trial_slope = self.linemodel.grad(self.step)
        return self.step
