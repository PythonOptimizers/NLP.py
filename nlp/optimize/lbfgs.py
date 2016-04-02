# -*- coding: utf-8 -*-
"""The limited-memory BFGS linesearch method for unconstrained optimization."""

import logging
from nlp.model.linemodel import C1LineModel
from nlp.ls.linesearch import ArmijoWolfeLineSearch
from nlp.tools import norms
from nlp.tools.exceptions import UserExitRequest
from nlp.tools.timing import cputime

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
        self.status = ""

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
        fmt_short = "%4d  %8.1e  %7.1e"
        fmt = fmt_short + "  %8.1e  %4d"
        ls_fmt = "%7.1e  %8.1e"

        tstart = cputime()

        self.f0 = f = model.obj(x)
        g = model.grad(x)
        self.gNorm0 = gNorm = norms.norm2(g)
        stoptol = max(self.abstol, self.reltol * self.gNorm0)

        exitUser = False
        exitOptimal = gNorm <= stoptol
        exitIter = self.iter >= self.maxiter
        status = ""

        while not (exitUser or exitOptimal or exitIter):

            # Obtain search direction
            H = model.hop(x)
            d = -(H * g)

            # Prepare for modified linesearch
            step0 = (1.0 / gNorm) if self.iter == 0 else 1.0
            line_model = C1LineModel(self.model, x, d)
            ls = ArmijoWolfeLineSearch(line_model, step=step0)
            for step in ls:
                self.logger.debug(ls_fmt, step, ls.trial_value)

            self.logger.info(fmt, self.iter, f, gNorm, ls.slope, ls.bk)

            # Prepare new pair {s,y} to be inserted into L-BFGS operator.
            self.s = ls.step * d
            x = ls.iterate
            g_next = line_model.gradval
            self.y = g_next - g
            status = ""
            try:
                self.post_iteration()
            except UserExitRequest:
                status = "usr"

            # Prepare for next round.
            g = g_next
            gNorm = norms.norm2(g)
            f = ls.trial_value
            self.iter += 1

            exitOptimal = gNorm <= stoptol
            exitIter = self.iter > self.maxiter
            exitUser = status == "usr"

        self.tsolve = cputime() - tstart
        self.logger.info(fmt_short, self.iter, f, gNorm)

        self.x = x
        self.f = f
        self.g = g
        self.gNorm = gNorm

        # Set final solver status.
        if status == "usr":
            pass
        elif self.gNorm <= stoptol:
            status = "opt"
        else:  # self.iter > self.maxiter:
            status = "itr"
        self.status = status
