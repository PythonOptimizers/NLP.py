"""Strong Wolfe linesearch."""

from math import sqrt
import numpy as np
from nlp.ls.linesearch import LineSearch, LineSearchFailure
from nlp.ls._strong_wolfe_linesearch import dcsrch


class StrongWolfeLineSearch(LineSearch):
    u"""Moré-Thuente linesearch for the strong Wolfe conditions.

         ϕ(t)   ≤ ϕ(0) + ftol * t * ϕ'(0)  (Armijo condition)
        |ϕ'(t)| ≤ gtol * |ϕ'(0)|           (curvature condition)

    where 0 < ftol < gtol < 1.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a strong Wolfe linesearch procedure.

        :keywords:
            :ftol: constant used in the Armijo condition (1.0e-4)
            :gtol: constant used in the curvature condition (0.9)
            :xtol: minimal relative step bracket length (1.0e-10)
            :lb: initial lower bound of the bracket
            :ub: initial upper bound of the bracket
        """
        super(StrongWolfeLineSearch, self).__init__(*args, **kwargs)
        sqeps = sqrt(np.finfo(np.double).eps)
        self.__ftol = max(min(kwargs.get("ftol", 1.0e-4),
                              1 - sqeps),
                          sqeps)
        self.__gtol = min(max(kwargs.get("gtol", 0.9),
                              self.ftol + sqeps),
                          1 - sqeps)
        self.__xtol = kwargs.get("xtol", 1.0e-10)
        self._lb = kwargs.get("lb", 0.0)
        self._ub = kwargs.get("ub",
                              max(4 * min(self.step, 1.0),
                                  -0.1 * self.value / self.slope / self.ftol))

        self._trial_slope = self.linemodel.grad(self.step, x=self.iterate)
        self.__task = "START"
        self.__isave = np.empty(2, dtype=np.int32)
        self.__dsave = np.empty(13, dtype=np.double)

        self._step, self.__task, self.__isave, self.__dsave = \
            dcsrch(self._step, self._value, self._slope,
                   self.ftol, self.gtol, self.xtol,
                   self.__task, self.lb, self.ub,
                   self.__isave, self.__dsave)

    @property
    def ftol(self):
        return self.__ftol

    @property
    def gtol(self):
        return self.__gtol

    @property
    def xtol(self):
        return self.__xtol

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def trial_slope(self):
        return self._trial_slope

    def next(self):
        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step, x=self.iterate)
        self._trial_slope = self.linemodel.grad(self.step, x=self.iterate)

        self._step, self.__task, self.__isave, self.__dsave = \
            dcsrch(self._step, self._trial_value, self._trial_slope,
                   self.ftol, self.gtol, self.xtol,
                   self.__task, self.lb, self.ub,
                   self.__isave, self.__dsave)

        if self.__task[:4] == "CONV":  # strong Wolfe conditions satisfied
            raise StopIteration()

        if self.__task[:2] != "FG":
            raise LineSearchFailure(self.__task)
