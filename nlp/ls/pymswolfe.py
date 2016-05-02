# -*- coding: utf-8 -*-
"""
PyMSWolfe: Jorge Nocedal's modified More and Thuente linesearch
guaranteeing satisfaction of the strong Wolfe conditions.
"""

import numpy as np
from nlp.ls._modified_strong_wolfe_linesearch import mcsrch


class StrongWolfeLineSearch:
    """
    A general-purpose linesearch procedure enforcing the strong
    Wolfe conditions

    f(x+td) <= f(x) + ftol * t * <g(x),d>    (Armijo condition)

    | <g(x+td),d> | <= gtol * | <g(x),d> |   (curvature condition)

    This is a Python interface to Jorge Nocedal's modification of
    the More and Thuente linesearch. Usage of this class is slightly
    different from the original More and Thuente linesearch.

    Instantiate as follows

    SWLS = StrongWolfeLineSearch(f, x, g, d, obj, grad)

    where

    * f     is the objective value at the current iterate x
    * x     is the current iterate
    * g     is the objective gradient at the current iterate x
    * d     is the current search direction
    * obj   is a scalar function used to evaluate the value of
            the objective at a given point
    * grad  is a scalar function used to evaluate the gradient
            of the objective at a given point.

    :keywords:

        :ftol:    the constant used in the Armijo condition (1e-4)
        :gtol:    the constant used in the curvature condition (0.9)
        :xtol:    a minimal relative step bracket length (1e-16)
        :stp:     an initial step value (1.0)
        :stpmin:  the initial lower bound of the bracket (1e-20)
        :stpmax:  the initial upper bound of the bracket (1e+20)
        :maxfev:  the maximum number of function evaluations permitted (20)

    To ensure existence of a step satisfying the strong Wolfe
    conditions, d should be a descent direction for f at x and
    ftol <= gtol.

    The final value of the step will be held in SWLS.stp

    After the search, SWLS.armijo will be set to True if the step
    computed satisfies the Armijo condition and SWLS.curvature will
    be set to True if the step satisfies the curvature condition.
    """

    def __init__(self, f, x, g, d, obj, grad, **kwargs):

        # Mandatory arguments
        self.f = f          # Function value f(xk)
        self.x = x.copy()   # xk
        self.g = g.copy()   # Gradient of f at xk
        self.d = d          # Direction along which to search
        self.n = self.g.shape[0]

        self.obj = obj   # To evaluate function value
        self.grad = grad  # To evaluate function gradient

        # Optional arguments
        self.ftol = kwargs.get('ftol', 1.0e-4)
        self.gtol = kwargs.get('gtol', 0.9)
        self.xtol = kwargs.get('xtol', 1.0e-16)
        self.stp = kwargs.get('stp', 1.0)
        self.stpmin = kwargs.get('stpmin', 1.0e-20)
        self.stpmax = kwargs.get('stpmax', 1.0e+20)
        self.maxfev = kwargs.get('maxfev', 20)

        self.armijo = False
        self.curvature = False
        self.info = 99
        self.nfev = 0
        self.wa = np.empty(self.n, dtype=np.float64)

    def search(self):

        isave = np.empty(2, dtype=np.int32)
        dsave = np.empty(21, dtype=np.double)
        self.x, self.f, self.g, self.stp, self.info, self.nfev, self.wa \
            = mcsrch(self.n, self.x, self.f, self.g, self.d, self.stp,
                     self.ftol, self.gtol, self.xtol, self.stpmin,
                     self.stpmax, self.maxfev, self.info, self.nfev,
                     isave, dsave, self.wa)

        while self.info == -1:
            self.f = self.obj(self.x)
            self.g = self.grad(self.x)
            self.x, self.f, self.g, self.stp, self.info, self.nfev, self.wa \
                = mcsrch(self.n, self.x, self.f, self.g, self.d, self.stp,
                         self.ftol, self.gtol, self.xtol, self.stpmin,
                         self.stpmax, self.maxfev, self.info, self.nfev,
                         isave, dsave, self.wa)

        if self.info == 1:     # Strong Wolfe conditions satisfied
            self.armijo = True
            self.curvature = True

        if self.info == 0:
            print ' linesearch returned; incorrect input values:'
            print '    stpmin = ', self.stpmin
            print '    stpmax = ', self.stpmax
            print '    ftol   = ', self.ftol
            print '    gtol   = ', self.gtol
            print '    xtol   = ', self.xtol
            print '    n      = ', self.n
            print '    stp    = ', self.stp
            print '    maxfev = ', self.maxfev
        return


if __name__ == '__main__':

    from nlp.model.amplpy import AmplModel
    from math import sqrt
    import sys

    model = AmplModel(sys.argv[1])
    f = model.obj(model.x0)
    g = model.grad(model.x0)
    d = -g
    SWLS = StrongWolfeLineSearch(f,
                                 model.x0,
                                 g,
                                 d,
                                 lambda z: model.obj(z),
                                 lambda z: model.grad(z),
                                 stp=1.0 / sqrt(np.dot(g, g)))
    print ' Before search'
    print '   f = ', f
    print '   stpmax = ', SWLS.stpmax
    SWLS.search()
    print ' After search'
    print '   f = ', SWLS.f
    print '   step length = ', SWLS.stp
