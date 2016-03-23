# -*- coding: utf-8 -*-
from pykrylov.linop.lbfgs import InverseLBFGSOperator as InverseLBFGS
from nlp.ls.pymswolfe import StrongWolfeLineSearch
from nlp.tools import norms
from nlp.tools.timing import cputime

__docformat__ = 'restructuredtext'


class LBFGSFramework(object):
    """Solve unconstrained problems with the limited-memory BFGS method.

    :keywords:
        :npairs:    the number of (s,y) pairs to store (default: 5)
        :maxiter:   the maximum number of iterations (default: max(10n, 1000))
        :atol:      absolute stopping tolerance (default: 1.0e-8)
        :rtol:      relative stopping tolerance (default: 1.0e-6)

    Other keyword arguments will be passed to InverseLBFGS.

    The linesearch used in this version is Jorge Nocedal's modified Moré and
    Thuente linesearch, attempting to ensure satisfaction of the strong Wolfe
    conditions. The modifications attempt to limit the effects of rounding
    error inherent to the More and Thuente linesearch.
    """
    def __init__(self, model, **kwargs):

        self.model = model
        self.npairs = kwargs.get("npairs", 5)
        self.abstol = kwargs.get("atol", 1.0e-8)
        self.reltol = kwargs.get("rtol", 1.0e-6)
        self.iter = 0
        self.nresets = 0
        self.converged = False

        self.lbfgs = InverseLBFGS(model.nvar, **kwargs)
        self.x = model.x0

        self.f = None
        self.g = None
        self.gNorm = None
        self.f0 = None
        self.gNorm0 = None

        self.tsolve = 0.0

    def solve(self, **kwargs):

        model = self.model
        self.maxiter = kwargs.get("maxiter", max(10 * model.nvar, 1000))

        tstart = cputime()

        self.f = self.model.obj(self.x)
        self.g = self.model.grad(self.x)
        self.gNorm = norms.norm2(self.g)
        self.f0 = self.f
        self.gNorm0 = self.gNorm

        stoptol = max(self.abstol, self.reltol * self.gNorm0)

        while self.gNorm > stoptol and self.iter < self.maxiter:

            if not self.silent:
                print '%-5d  %-12g  %-12g' % (self.iter, self.f, self.gNorm)

            # Obtain search direction
            d = self.lbfgs * (-self.g)

            # Prepare for modified More-Thuente linesearch
            if self.iter == 0:
                stp0 = 1.0 / self.gNorm
            else:
                stp0 = 1.0
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.x,
                                         self.g,
                                         d,
                                         lambda z: self.model.obj(z),
                                         lambda z: self.model.grad(z),
                                         stp=stp0)
            # Perform linesearch
            SWLS.search()

            # SWLS.x  contains the new iterate
            # SWLS.g  contains the objective gradient at the new iterate
            # SWLS.f  contains the objective value at the new iterate
            s = SWLS.x - self.x
            self.x = SWLS.x
            y = SWLS.g - self.g
            self.g = SWLS.g
            self.gNorm = norms.norm2(self.g)
            self.f = SWLS.f

            # Update inverse Hessian approximation using the most recent pair
            self.lbfgs.store(s, y)
            self.iter += 1

        self.tsolve = cputime() - tstart
        self.converged = (self.iter < self.maxiter)
