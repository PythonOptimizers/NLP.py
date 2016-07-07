# -*- coding: utf8 -*-
"""Quadratic/Cubic linesearch."""

from math import sqrt
import numpy as np
from nlp.ls.linesearch import LineSearch, LineSearchFailure

eps = np.finfo(np.double).eps
sqeps = sqrt(eps)


class QuadraticCubicLineSearch(LineSearch):
    u"""Quadratic/cubic linesearch satisfying Armijo condition.

         ϕ(t) ≤ ϕ(0) + ftol * t * ϕ'(0)  (Armijo condition)
    where 0 < ftol < 1.
    """

    def __init__(self, *args, **kwargs):
        u"""Instantiate a linesearch using quadratic or cubic interpolants.

        The search stops as soon as a step size t is found such that

            ϕ(t) ≤ ϕ(0) + t * ftol * ϕ'(0)

        where 0 < ftol < 1 and ϕ'(0) is the directional derivative of
        a merit function f in the descent direction d.

        :keywords:
            :ftol: constant used in Armijo condition (default: 1.0e-4)
            :bkmax: maximum number of backtracking steps (default: 20)
        """
        name = kwargs.pop("name", "Armijo linesearch")
        super(QuadraticCubicLineSearch, self).__init__(*args, name=name,
                                                       **kwargs)
        self.__ftol = max(min(kwargs.get("ftol", 1.0e-4), 1 - sqeps), sqeps)
        self.__bkmax = max(kwargs.get("bkmax", 20), 0)
        self.__eps1 = self.__eps2 = sqrt(eps) * 100
        self._bk = 0
        self._last_step = None
        self._last_trial_value = None
        return

    @property
    def ftol(self):
        return self.__ftol

    @property
    def bkmax(self):
        return self.__bkmax

    @property
    def bk(self):
        return self._bk

    def next(self):
        if self.trial_value <= self.value + self.step * self.ftol * self.slope:
            raise StopIteration()

        self._bk += 1
        if self._bk > self.__bkmax:
            raise LineSearchFailure("backtracking limit exceeded")

        if self.bk == 1:
            # quadratic interpolation
            # ϕq(t) = t²(ϕ(t₀) - ϕ(0) - t₀ϕ'(0))/t₀² + ϕ'(0)t + ϕ(0)
            self._last_step = self.step
            self._last_trial_value = self.trial_value
            step = -self.slope * self._last_step**2
            step /= 2 * (self.trial_value -
                         self.value - self.slope * self._last_step)
        else:
            # cubic interpolation
            a0 = self._last_step
            a1 = self.step
            phi0 = self.value
            phi_a0 = self._last_trial_value
            phi_a1 = self.trial_value
            c = 1. / (a0**2 * a1**2 * (a1 - a0))
            M = np.array([[a0**2, -a1**2], [-a0**3, a1**3]])
            coeffs = c * np.dot(M, np.array([phi_a1 - phi0 - self.slope * a1,
                                             phi_a0 - phi0 - self.slope * a0]))
            step = -coeffs[1]
            step += sqrt(coeffs[1]**2 - 3 * coeffs[0] * self.slope)
            step /= 3 * coeffs[0]

            self._last_step = self.step
            self._last_trial_value = self.trial_value

        if abs(step - self._step) < self.__eps1 or abs(step) < self.__eps2:
            self._step = self.step / 2
        else:
            self._step = step

        if self.step < self.stepmin:
            raise LineSearchFailure("linesearch step too small")

        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step, x=self.iterate)

        return self.step  # return value of step just tested
