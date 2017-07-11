# -*- coding: utf-8 -*-
"""Auglag: Bound-Constrained augmented Lagrangian method.

For references on these methods see
    [CGT91] A. R. Conn, N. I. M. Gould, and Ph. L. Toint, *LANCELOT: A Fortran
            Package for Large-Scale Nonlinear Optimization (Release A)*,
            Springer-Verlag, 1992
    [NoW06] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd Edition
            Springer, 2006, pp 519--523.
"""
import logging
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.model.augmented_lagrangian import QuasiNewtonAugmentedLagrangian
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.exceptions import UserExitRequest
from nlp.tools.utils import project, where
from nlp.tools.timing import cputime

from pykrylov.lls.lsqr import LSQRFramework as LSQRSolver
from pykrylov.linop import ReducedLinearOperator as ReducedJacobian

import numpy as np

__docformat__ = "restructuredtext"


class Auglag(object):
    """Bound-Constrained Augmented Lagrangian solver."""

    def __init__(self, model, bc_solver, **kwargs):
        u"""Instantiate an augmented Lagrangian solver for general constrained problem.

        The model should have the general form

            min f(x)  subject to   cₗ ≤ c(x) ≤ cᵤ,    l ≤ x ≤ u.

        The augmented Lagrangian is defined as:

            L(x, π; ρ) := f(x) - π"c(x) + ½ ρ |c(x)|².
        where π are the current Lagrange multiplier estimates and ρ is the
        current penalty parameter.

        The algorithm stops as soon as the infinity norm of the projected
        gradient of the Lagrangian falls below ``max(abstol, reltol * pg0)``
        where ``pg0`` is the infinity norm of the projected gradient of the
        Lagrangian at the initial point.

        :parameters:
            :model:            a :class:`NLPModel` object representing the
                               problem. For instance, model may arise from an
                               AMPL model
            :bc_solver:        a solver for solving the inner iteration
                               subproblem
        :keywords:
            :x0:               starting point                  (`model.x0`)
            :reltol:           relative stopping tolerance     (1.0e-5)
            :abstol:           absolute stopping tolerance     (1.0e-12)
            :maxiter:          maximum number of iterations    (max(1000, 10n))
            :maxupdate:        maximum number of penalty or multiplier
                               updates                         (100)
            :ny:               apply Nocedal/Yuan linesearch   (False)
            :nbk:              max number of backtracking steps in Nocedal/Yuan
                               linesearch                      (5)
            :monotone:         use monotone descent strategy   (False)
            :n_iter_non_mono:  number of iterations for which non-strict
                               descent can be tolerated if monotone = False
                                                               (25)
            :least_squares_pi: initialize with least squares multipliers (True)
            :logger_name:      name of a logger object that can be used in the
                               post-iteration                  (nlp.auglag)

        :Exit codes:
            :opt:    Optimal solution found
            :iter:   Maximum iteration reached
            :feas:   Feasible, but not optimal, solution found
            :fail:   Cannot make further progress from current point
            :stal:   Problem converged to an infeasible point
            :time:   Time limit exceeded
        """
        full_qn = kwargs.get("full_qn",False)
        if full_qn:
            self.model = QuasiNewtonAugmentedLagrangian(model, **kwargs)
        else:
            self.model = AugmentedLagrangian(model, **kwargs)

        print self.model
        print self.model.model

        self.x = kwargs.get("x0", self.model.x0.copy())

        self.least_squares_pi = kwargs.get("least_squares_pi", True)

        self.bc_solver = bc_solver

        self.tau = kwargs.get("tau", 0.1)
        self.omega = None
        self.eta = None
        self.eta0 = 0.1258925
        self.omega0 = 1.
        self.omega_init = kwargs.get(
            "omega_init", self.omega0 * 0.1)  # penalty_init**-1
        self.eta_init = kwargs.get(
            "eta_init", self.eta0**0.1)  # penalty_init**-0.1
        self.a_omega = kwargs.get("a_omega", 1.)
        self.b_omega = kwargs.get("b_omega", 1.)
        self.a_eta = kwargs.get("a_eta", 0.1)
        self.b_eta = kwargs.get("b_eta", 0.9)
        self.omega_rel = kwargs.get("omega_rel", 1.e-5)
        self.omega_abs = kwargs.get("omega_abs", 1.e-7)
        self.eta_rel = kwargs.get("eta_rel", 1.e-5)
        self.eta_abs = kwargs.get("eta_abs", 1.e-7)

        self.f0 = self.f = None

        # Maximum number of inner iterations
        self.maxiter = kwargs.get("maxiter",
                                  100 * self.model.model.original_n)

        self.maxupdate = kwargs.get("maxupdate",100)

        # Maximum run time
        self.maxtime = kwargs.get("maxtime", 3600.)

        self.update_on_rejected_step = False

        self.inner_fail_count = 0
        self.status = None

        self.hformat = "%-5s  %8s  %8s  %8s  %8s  %5s  %4s  %8s  %8s"
        self.header = self.hformat % ("iter", "f", u"‖P∇L‖", u"‖c‖", u"ρ",
                                      "inner", "stat", u"ω", u"η")
        self.format = "%-5d %8.1e %8.1e %8.1e %8.1e %5d %4s %8.1e %8.1e"
        self.format0 = "%-5d %8.1e %8.1e %8s %8s %5s %4s %8.1e %8.1e"

        # Initialize some counters for counting number of Hprod used in
        # BQP linesearch and CG.
        self.hprod_bqp_linesearch = 0
        self.hprod_bqp_linesearch_fail = 0
        self.nlinesearch = 0
        self.hprod_bqp_cg = 0
        self.tsolve = 0.0

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get("logger_name", "nlp.auglag")
        self.log = logging.getLogger(logger_name)
        self.log.propagate = False

    def project_gradient(self, x, g):
        """Project the provided gradient into the bounds.

        This is a helper function for determining optimality conditions of the
        original NLP.
        """
        p = x - g
        med = np.maximum(np.minimum(p, self.model.Uvar), self.model.Lvar)
        q = x - med
        return q

    def get_active_bounds(self, x, l, u):
        """Return a list of indices of variables that are at a bound."""
        lower_active = where(x == l)
        upper_active = where(x == u)
        active_bound = np.concatenate((lower_active, upper_active))
        return active_bound

    def least_squares_multipliers(self, x):
        """Compute least-squares multipliers estimates."""
        al_model = self.model
        slack_model = self.model.model
        m = slack_model.m
        n = slack_model.n

        lim = max(2 * m, 2 * n)
        J = slack_model.jop(x)

        # Determine which bounds are active to remove appropriate columns of J
        on_bound = self.get_active_bounds(x,
                                          slack_model.Lvar,
                                          slack_model.Uvar)
        free_vars = np.setdiff1d(np.arange(n, dtype=np.int), on_bound)
        Jred = ReducedJacobian(J, np.arange(m, dtype=np.int),
                               free_vars)

        g = slack_model.grad(x) - J.T * al_model.pi

        lsqr = LSQRSolver(Jred.T)
        lsqr.solve(g[free_vars], itnlim=lim)
        if lsqr.optimal:
            al_model.pi += lsqr.x.copy()
        else:
            self.log.debug("lsqr failed to converge")
        return

    def update_multipliers(self, convals, status):
        """Update multipliers and tighten tolerances."""
        # TODO: refactor this
        al_model = self.model
        slack_model = self.model.model

        if self.least_squares_pi:
            self.least_squares_multipliers(self.x)
        else:
            al_model.pi -= al_model.penalty * convals

        if slack_model.m != 0:
            self.log.debug("New multipliers = %g, %g" %
                           (max(al_model.pi), min(al_model.pi)))

        if status == "gtol":
            # Safeguard: tighten tolerances only if desired optimality
            # is reached to prevent rapid decay of the tolerances from failed
            # inner loops
            self.eta /= al_model.penalty**self.b_eta
            self.omega /= al_model.penalty**self.b_omega
            self.inner_fail_count = 0
        else:
            self.inner_fail_count += 1
        return

    def update_penalty_parameter(self):
        """Increase penalty parameter and reset tolerances.

        Tolerances are reset based on new penalty value.
        """
        al_model = self.model
        al_model.penalty /= self.tau
        self.eta = self.eta0 * al_model.penalty**-self.a_eta
        self.omega = self.omega0 * al_model.penalty**-self.a_omega
        return

    def post_iteration(self, **kwargs):
        """Perform post-iteration updates.

        Override this method to perform additional work at the end of a
        major iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None

    def setup_bc_solver(self):
        """Setup bound-constrained solver."""
        return self.bc_solver(self.model, TruncatedCG, greltol=self.omega,
                              x0=self.x)

    def solve(self, **kwargs):
        """Solve method.

        All keyword arguments are passed directly to the constructor of the
        bound constraint solver.
        """
        al_model = self.model
        slack_model = self.model.model

        on = slack_model.original_n

        # Move starting point into the feasible box
        self.x = project(self.x, al_model.Lvar, al_model.Uvar)

        # "Smart" initialization of slack variables using the magical step
        # function that is already available
        (self.x, m_step_init) = self.model.magical_step(self.x)

        dL = al_model.dual_feasibility(self.x)
        self.f = self.f0 = self.model.model.model.obj(self.x[:on])

        PdL = self.project_gradient(self.x, dL)
        Pmax = np.max(np.abs(PdL))
        self.pg0 = self.pgnorm = Pmax

        # Specific handling for the case where the original NLP is
        # unconstrained
        if slack_model.m == 0:
            max_cons = 0.
        else:
            max_cons = np.max(np.abs(slack_model.cons(self.x)))
            cons_norm_ref = max_cons

        self.cons0 = max_cons

        self.omega = self.omega_init
        self.eta = self.eta_init
        self.omega_opt = self.omega_rel * self.pg0 + self.omega_abs
        self.eta_opt = self.eta_rel * max_cons + self.eta_abs

        self.iter = 0
        self.inner_fail_count = 0
        self.niter_total = 0
        infeas_iter = 0

        exitIter = False
        exitTime = False
        # Convergence check
        exitOptimal = (Pmax <= self.omega_opt and max_cons <= self.eta_opt)
        if exitOptimal:
            self.status = "opt"

        tick = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0:
            self.log.info(self.header)
            self.log.info(self.format0, self.iter, self.f, self.pg0,
                          self.cons0, al_model.penalty, "", "", self.omega,
                          self.eta)

        while not (exitOptimal or exitIter or exitTime):
            self.iter += 1

            # Perform bound-constrained minimization
            # TODO: set appropriate stopping conditions
            bc_solver = self.setup_bc_solver()
            bc_solver.solve()
            self.x = bc_solver.x.copy()  # may not be useful.
            self.niter_total += bc_solver.iter + 1

            dL = al_model.dual_feasibility(self.x)
            PdL = self.project_gradient(self.x, dL)
            Pmax = np.max(np.abs(PdL))
            convals = slack_model.cons(self.x)

            # Specific handling for the case where the original NLP is
            # unconstrained
            if slack_model.m == 0:
                max_cons = 0.
            else:
                max_cons = np.max(np.abs(convals))

            self.f = self.model.model.model.obj(self.x[:on])
            self.pgnorm = Pmax

            # Print out header, say, every 20 iterations.
            if self.iter % 20 == 0:
                self.log.info(self.header)

            self.log.info(self.format % (self.iter, self.f,
                                         self.pgnorm, max_cons,
                                         al_model.penalty,
                                         bc_solver.iter, bc_solver.status,
                                         self.omega, self.eta))

            # Update penalty parameter or multipliers based on result
            if max_cons <= np.maximum(self.eta, self.eta_opt):

                # Update convergence check
                if max_cons <= self.eta_opt and Pmax <= self.omega_opt:
                    exitOptimal = True
                    break

                self.update_multipliers(convals, bc_solver.status)

                # Update reference constraint norm on successful reduction
                cons_norm_ref = max_cons
                infeas_iter = 0

                # If optimality of the inner loop is not achieved within 10
                # major iterations, exit immediately
                if self.inner_fail_count == 10:
                    if max_cons <= self.eta_opt:
                        self.status = "feas"
                        self.log.debug("cannot improve current point, exiting")
                    else:
                        self.status = "fail"
                        self.log.debug("cannot improve current point, exiting")
                    break

                self.log.debug("updating multipliers estimates")

            else:

                self.update_penalty_parameter()
                self.log.debug("keeping current multipliers estimates")

                if max_cons > 0.99 * cons_norm_ref and self.iter != 1:
                    infeas_iter += 1
                else:
                    cons_norm_ref = max_cons
                    infeas_iter = 0

                if infeas_iter == 10:
                    self.status = "stal"
                    self.log.debug("problem appears infeasible, exiting")
                    break

            # Safeguard: tightest tolerance should be near optimality to
            # prevent excessive inner loop iterations at the end of the
            # algorithm
            if self.omega < self.omega_opt:
                self.omega = self.omega_opt
            if self.eta < self.eta_opt:
                self.eta = self.eta_opt

            try:
                self.post_iteration()
            except UserExitRequest:
                self.status = "usr"

            exitIter = self.niter_total > self.maxiter or self.iter > self.maxupdate

            exitTime = (cputime() - tick) > self.maxtime

        self.tsolve = cputime() - tick    # Solve time

        # Solution output, etc.
        if exitOptimal:
            self.status = "opt"
            self.log.debug("optimal solution found")
        elif not exitOptimal and exitTime:
            self.status = "time"
            self.log.debug("maximum run time exceeded")
        elif not exitOptimal and exitIter:
            self.status = "iter"
            self.log.debug("maximum number of iterations reached")

        self.log.info("f = %12.8g" % self.f)
        if slack_model.m != 0:
            self.log.info("pi_max = %12.8g" % np.max(al_model.pi))
            self.log.info("max infeas. = %12.8g" % max_cons)
