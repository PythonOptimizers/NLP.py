# -*- coding: utf-8 -*-
"""The limited-memory BFGS linesearch method for unconstrained optimization."""

from nlp.model.linemodel import C1LineModel
from nlp.ls.linesearch import ArmijoWolfeLineSearch
from nlp.tools import norms
from nlp.tools.timing import cputime
import logging

__docformat__ = 'restructuredtext'


class LBFGSFramework(object):
    """Solve unconstrained problems with the limited-memory BFGS method."""

    def __init__(self, model, **kwargs):
        """Instantiate a L-BFGS solver for ``model``.

        :parameters:
            :model: a ``QuasiNewtonModel`` based on ``InverseLBFGSOperator``

        :keywords:
            :maxiter: maximum number of iterations (default: max(10n, 1000))
            :atol: absolute stopping tolerance (default: 1.0e-8)
            :rtol: relative stopping tolerance (default: 1.0e-6)
            :logger_name: name of a logger (default: 'nlp.lbfgs')
        """
        self.model = model
        self.maxiter = kwargs.get("maxiter", max(10 * model.nvar, 1000))
        self.abstol = kwargs.get("atol", 1.0e-8)
        self.reltol = kwargs.get("rtol", 1.0e-6)

        logger_name = kwargs.get("logger_name", "nlp.lbfgs")
        self.logger = logging.getLogger(logger_name)

        self.iter = 0
        self.converged = False

        self.x = model.x0.copy()
        self.f = None
        self.g = None
        self.gNorm = None
        self.f0 = None
        self.gNorm0 = None

        self.s = None
        self.y = None

        self.tsolve = 0.0

    def post_iteration(self):
        """Bookkeeping at the end of a general iteration."""
        self.model.H.store(self.s, self.y)

    def solve(self):
        """Solve model with the L-BFGS method."""
        model = self.model
        x = self.x
        hdr = "%4s  %8s  %7s  %8s  %4s" % ("iter", "f", u"‖∇f‖", u"∇f'd", "bk")
        self.logger.info(hdr)
        fmt = "%4d  %8.1e  %7.1e"
        tstart = cputime()

        self.f0 = f = model.obj(x)
        g = model.grad(x)
        self.gNorm0 = gNorm = norms.norm2(g)

        info = fmt % (self.iter, f, gNorm)
        stoptol = max(self.abstol, self.reltol * self.gNorm0)

        while gNorm > stoptol and self.iter < self.maxiter:

            # Obtain search direction
            H = model.hop(x)
            d = -(H * g)

            # Prepare for modified linesearch
            step0 = (1.0 / gNorm) if self.iter == 0 else 1.0
            line_model = C1LineModel(self.model, x, d)
            ls = ArmijoWolfeLineSearch(line_model, step=step0)
            for step in ls:
                self.logger.debug("%7.1e  %8.1e" % (step, ls.trial_value))

            info += "  %8.1e  %4d" % (ls.slope, ls.bk)
            self.logger.info(info)

            # Prepare new pair {s,y} to be inserted into L-BFGS operator.
            self.s = ls.step * d
            x = ls.iterate
            g_next = line_model.gradval
            self.y = g_next - g
            self.post_iteration()

            # Prepare for next round.
            g = g_next
            gNorm = norms.norm2(g)
            f = ls.trial_value
            self.iter += 1
            info = fmt % (self.iter, f, gNorm)

        self.tsolve = cputime() - tstart
        self.logger.info(info)
        self.x = x
        self.f = f
        self.g = g
        self.gNorm = gNorm
        self.converged = (self.iter < self.maxiter)
