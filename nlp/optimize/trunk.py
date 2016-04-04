# -*- coding: utf-8 -*-
"""TRUNK: Trust-Region Method for Unconstrained Programming.

D. Orban            Montreal Sept. 2003
"""
from nlp.model.nlpmodel import QPModel
from nlp.tr.trustregion import TrustRegionSolver
from nlp.tools import norms
from nlp.tools.timing import cputime
from nlp.tools.exceptions import UserExitRequest
import numpy as np
import logging
from math import sqrt

__docformat__ = "restructuredtext"


class Trunk(object):
    u"""Abstract trust-region method for unconstrained optimization.

    A stationary point of the unconstrained problem

        minimize f(x)

    is identified by solving a sequence of trust-region constrained quadratic
    subproblems

        min  gᵀs + ½ s'Hs  subject to  ‖s‖ ≤ Δ.
    """

    def __init__(self, nlp, tr, tr_solver, **kwargs):
        """Instantiate a trust-region solver for ``nlp``.

        :parameters:
            :nlp:       a :class:`NLPModel` instance.
            :tr:        a :class:`TrustRegion` instance.
            :tr_solver: a trust-region solver to be passed as argument to
                        the :class:`TrustRegionSolver` constructor.

        :keywords:
            :x0:           starting point                     (``nlp.x0``)
            :reltol:       relative stopping tolerance        (``nlp.stop_d``)
            :abstol:       absolute stopping tolerance        (1.0e-6)
            :maxiter:      maximum number of iterations       (max(1000,10n))
            :inexact:      use inexact Newton stopping tol    (``False``)
            :ny:           apply Nocedal/Yuan linesearch      (``False``)
            :nbk:          max number of backtracking steps in Nocedal/Yuan
                           linesearch                         (5)
            :monotone:     use monotone descent strategy      (``False``)
            :n_non_monotone: number of iterations for which non-strict descent
                           is tolerated if ``monotone=False`` (25)
            :logger_name:  name of a logger object that can be used in the post
                           iteration                          (``None``)

        Once a ``Trunk`` object has been instantiated and the problem is
        set up, solve problem by issuing a call to ``TRNK.solve()``.
        The algorithm stops as soon as the Euclidian norm of the gradient falls
        below

            ``max(abstol, reltol * g0)``

        where ``g0`` is the Euclidian norm of the initial gradient.
        """
        self.nlp = nlp
        self.tr = tr
        self.tr_solver = tr_solver
        self.solver = None  # Will point to subproblem solver data in Solve()
        self.iter = 0  # Iteration counter
        self.total_cgiter = 0
        self.x = kwargs.get("x0", self.nlp.x0.copy())
        self.f = None
        self.f0 = None
        self.g = None
        self.g_old = None
        self.save_g = False
        self.gNorm = None
        self.g0 = None
        self.alpha = 1.0  # For Nocedal-Yuan backtracking linesearch
        self.tsolve = 0.0

        self.step_accepted = False
        self.dvars = None
        self.dgrad = None
        self.status = ""

        self.reltol = kwargs.get("reltol", self.nlp.stop_d)
        self.abstol = kwargs.get("abstol", 1.0e-6)
        self.maxiter = kwargs.get("maxiter", max(1000, 10 * self.nlp.n))

        self.ny = kwargs.get("ny", False)
        self.nbk = kwargs.get("nbk", 5)
        self.inexact = kwargs.get("inexact", False)
        self.monotone = kwargs.get("monotone", False)
        self.n_non_monotone = kwargs.get("n_non_monotone", 25)
        self.logger = kwargs.get("logger", None)

        self.hformat = "%-5s %8s %7s %5s %8s %7s %7s %4s"
        self.header = self.hformat % ("iter", "f", u"‖∇f‖", "inner", u"ρ",
                                      u"‖step‖", "radius", "stat")
        self.hlen = len(self.header)
        self.format = "%-5d %8.1e %7.1e %5d %8.1e %7.1e %7.1e %4s"
        self.format0 = "%-5d %8.1e %7.1e %5s %8s %7s %7.1e %4s"
        self.radii = [tr.radius]

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get("logger_name", "nlp.trunk")
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False

    def precon(self, v, **kwargs):
        """Generic preconditioning method---must be overridden."""
        return v

    def post_iteration(self, **kwargs):
        """Perform work at the end of an iteration.

        Use this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def solve(self, **kwargs):
        """Solve.

        :keywords:
          :maxiter:  maximum number of iterations.

        All other keyword arguments are passed directly to the constructor of
        the trust-region solver.
        """
        nlp = self.nlp

        # Gather initial information.
        self.f = self.nlp.obj(self.x)
        self.f0 = self.f
        self.g = self.nlp.grad(self.x)
        self.g_old = self.g
        self.gNorm = norms.norm2(self.g)
        self.g0 = self.gNorm

        self.tr.radius = min(max(0.1 * self.gNorm, 1.0), 100)
        cgtol = 1.0 if self.inexact else -1.0
        stoptol = max(self.abstol, self.reltol * self.g0)
        step_status = None
        exitUser = False
        exitOptimal = self.gNorm <= stoptol
        exitIter = self.iter >= self.maxiter
        status = ""

        # Initialize non-monotonicity parameters.
        if not self.monotone:
            fMin = fRef = fCan = self.f0
            l = 0
            sigRef = sigCan = 0

        t = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0:
            self.log.info(self.header)
            self.log.info(self.format0,
                          self.iter, self.f, self.gNorm, "",
                          "", "", self.tr.radius, "")

        while not (exitUser or exitOptimal or exitIter):

            self.iter += 1
            self.alpha = 1.0

            if self.save_g:
                self.g_old = self.g.copy()

            # Iteratively minimize the quadratic model in the trust region
            # min m(s) := g's + ½ s'Hs
            # Note that m(s) does not include f(x): m(0) = 0.

            if self.inexact:
                cgtol = max(stoptol, min(0.7 * cgtol, 0.01 * self.gNorm))

            qp = QPModel(self.g, self.nlp.hop(self.x, self.nlp.pi0))
            self.solver = TrustRegionSolver(qp, self.tr_solver)
            self.solver.solve(prec=self.precon,
                              radius=self.tr.radius,
                              reltol=cgtol)

            step = self.solver.step
            snorm = self.solver.step_norm
            cgiter = self.solver.niter

            # Obtain model value at next candidate
            m = self.solver.m
            if m is None:
                m = qp.obj(step)

            self.total_cgiter += cgiter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)

            rho = self.tr.ratio(self.f, f_trial, m)

            if not self.monotone:
                rhoHis = (fRef - f_trial) / (sigRef - m)
                rho = max(rho, rhoHis)

            step_status = "Rej"
            self.step_accepted = False

            if rho >= self.tr.eta1:

                # Trust-region step is accepted.

                self.tr.update_radius(rho, snorm)
                self.x = x_trial
                self.f = f_trial
                self.g = nlp.grad(self.x)
                self.gNorm = norms.norm2(self.g)
                self.dvars = step
                if self.save_g:
                    self.dgrad = self.g - self.g_old
                step_status = "Acc"
                self.step_accepted = True

                # Update non-monotonicity parameters.
                if not self.monotone:
                    sigRef = sigRef - m
                    sigCan = sigCan - m
                    if f_trial < fMin:
                        fCan = f_trial
                        fMin = f_trial
                        sigCan = 0
                        l = 0
                    else:
                        l = l + 1

                    if f_trial > fCan:
                        fCan = f_trial
                        sigCan = 0

                    if l == self.n_non_monotone:
                        fRef = fCan
                        sigRef = sigCan

            else:

                # Trust-region step is rejected.
                if self.ny:  # Backtracking linesearch a la Nocedal & Yuan.

                    slope = np.dot(self.g, step)
                    bk = 0
                    armijo = self.f + 1.0e-4 * self.alpha * slope
                    while bk < self.nbk and f_trial >= armijo:
                        bk = bk + 1
                        self.alpha /= 1.2
                        x_trial = self.x + self.alpha * step
                        f_trial = nlp.obj(x_trial)
                    self.x = x_trial
                    self.f = f_trial
                    self.g = nlp.grad(self.x)
                    self.gNorm = norms.norm2(self.g)
                    self.tr.radius = self.alpha * snorm
                    snorm /= self.alpha
                    step_status = "N-Y"
                    self.step_accepted = True
                    self.dvars = self.alpha * step
                    if self.save_g:
                        self.dgrad = self.g - self.g_old

                else:
                    self.tr.update_radius(rho, snorm)

            self.step_status = step_status
            self.radii.append(self.tr.radius)
            status = ""
            try:
                self.post_iteration()
            except UserExitRequest:
                status = "usr"

            # Print out header, say, every 20 iterations
            if self.iter % 20 == 0:
                self.log.info(self.header)

            pstatus = step_status if step_status != "Acc" else ""
            self.log.info(self.format % (self.iter, self.f, self.gNorm, cgiter,
                                         rho, snorm, self.tr.radius, pstatus))

            exitOptimal = self.gNorm <= stoptol
            exitIter = self.iter > self.maxiter
            exitUser = status == "usr"

        self.tsolve = cputime() - t  # Solve time

        # Set final solver status.
        if status == "usr":
            pass
        elif self.gNorm <= stoptol:
            status = "opt"
        else:  # self.iter > self.maxiter:
            status = "itr"
        self.status = status


class QNTrunk(Trunk):
    def __init__(self, *args, **kwargs):
        super(QNTrunk, self).__init__(*args, **kwargs)
        self.save_g = True

    def post_iteration(self, **kwargs):
        # Update quasi-Newton approximation.
        if self.step_accepted:
            self.nlp.H.store(self.dvars, self.dgrad)
