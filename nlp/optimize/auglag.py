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

            L(x, π; ρ) := f(x) - π'c(x) + ½ ρ |c(x)|².
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
            :ny:               apply Nocedal/Yuan linesearch   (False)
            :nbk:              max number of backtracking steps in Nocedal/Yuan
                               linesearch                      (5)
            :monotone:         use monotone descent strategy   (False)
            :n_iter_non_mono:  number of iterations for which non-strict
                               descent can be tolerated if monotone = False
                                                               (25)
            :least_squares_pi: use of least squares to initialize Lagrange
                               multipliers                     (False)
            :logger_name:      name of a logger object that can be used in the
                               post-iteration                  (None)
            :verbose:          print log if True               (True)

        :Exit codes:
            :opt:    Optimal solution found
            :iter:   Maximum iteration reached
            :stal:   Not making sufficient progress
            :time:   Time limit exceeded
        """
        self.model = AugmentedLagrangian(model, **kwargs)
        print self.model
        print self.model.model

        self.x = kwargs.get('x0', self.model.x0.copy())

        self.least_squares_pi = kwargs.get('least_squares_pi', False)

        self.bc_solver = bc_solver

        self.tau = kwargs.get('tau', 0.1)
        self.omega = None
        self.eta = None
        self.eta0 = 0.1258925
        self.omega0 = 1.
        self.omega_init = kwargs.get(
            'omega_init', self.omega0 * 0.1)  # rho_init**-1
        self.eta_init = kwargs.get('eta_init', self.eta0**0.1)  # rho_init**-0.1
        self.a_omega = kwargs.get('a_omega', 1.)
        self.b_omega = kwargs.get('b_omega', 1.)
        self.a_eta = kwargs.get('a_eta', 0.1)
        self.b_eta = kwargs.get('b_eta', 0.9)
        self.omega_rel = kwargs.get('omega_rel', 1.e-5)
        self.omega_abs = kwargs.get('omega_abs', 1.e-7)
        self.eta_rel = kwargs.get('eta_rel', 1.e-5)
        self.eta_abs = kwargs.get('eta_abs', 1.e-7)

        self.f0 = self.f = None

        # Maximum number of total inner iterations
        self.max_inner_iter = kwargs.get('max_inner_iter',
                                         100 * self.model.model.original_n)

        self.update_on_rejected_step = False

        self.inner_fail_count = 0
        self.status = None

        self.verbose = kwargs.get('verbose', True)
        self.hformat = '%-5s  %8s  %8s  %8s  %8s  %5s  %4s  %8s  %8s'
        self.header = self.hformat % ("iter", "f", u"‖P∇L‖", u"‖c‖", u"ρ",
                                      "inner", "stat", u"ω", u"η")
        self.format = "%-5d  %8.1e  %8.1e  %8.1e  %8.1e  %5d  %4s  %8.1e  %8.1e"
        self.format0 = '%-5d  %8.1e  %8.1e  %8s  %8s  %5s  %4s  %8.1e  %8.1e'

        # Initialize some counters for counting number of Hprod used in
        # BQP linesearch and CG.
        self.hprod_bqp_linesearch = 0
        self.hprod_bqp_linesearch_fail = 0
        self.nlinesearch = 0
        self.hprod_bqp_cg = 0

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlp.auglag')
        self.log = logging.getLogger(logger_name)
        if not self.verbose:
            self.log.propagate = False

    def project_gradient(self, x, g):
        """
        Project the provided gradient on to the bound - constrained space and
        return the result. This is a helper function for determining
        optimality conditions of the original NLP.
        """
        p = x - g
        med = np.maximum(np.minimum(p, self.model.Uvar), self.model.Lvar)
        q = x - med
        return q

    def magical_step(self, x, g):
        """
        Compute a "magical step" to improve the convergence rate of the
        inner minimization algorithm. This step minimizes the augmented
        Lagrangian with respect to the slack variables only for a fixed set
        of decision variables.
        """
        al_model = self.model
        slack_model = self.model.model
        on = slack_model.original_n
        m_step = np.zeros(al_model.n)
        m_step[on:] = -g[on:] / al_model.rho
        # Assuming slack variables are restricted to [0,+inf) interval
        m_step[on:] = np.where(-m_step[on:] > x[on:], -x[on:], m_step[on:])
        return m_step

    def get_active_bounds(self, x, l, u):
        """
        Returns a list containing the indices of variables that are at
        either their lower or upper bound.
        """
        lower_active = where(x == l)
        upper_active = where(x == u)
        active_bound = np.concatenate((lower_active, upper_active))
        return active_bound

    def least_squares_multipliers(self, x):
        """
        Compute a least-squares estimate of the Lagrange multipliers for the
        current point. This may lead to faster convergence of the augmented
        Lagrangian algorithm, at the expense of more Jacobian-vector products.
        """
        slack_model = self.model.model
        m = slack_model.m
        n = slack_model.n

        lim = max(2 * m, 2 * n)
        J = slack_model.jac(x)

        # Determine which bounds are active to remove appropriate columns of J
        on_bound = self.get_active_bounds(x,
                                          slack_model.Lvar,
                                          slack_model.Uvar)
        free_vars = np.setdiff1d(np.arange(n, dtype=np.int), on_bound)
        Jred = ReducedJacobian(J, np.arange(m, dtype=np.int),
                               free_vars)

        g = slack_model.grad(x)

        # Call LSQR method
        lsqr = LSQRSolver(Jred.T)
        lsqr.solve(g[free_vars], itnlim=lim)
        if lsqr.optimal:
            self.pi = lsqr.x.copy()
        return

    def update_multipliers(self, convals, status):
        """
        Infeasibility is sufficiently small; update multipliers and
        tighten feasibility and optimality tolerances
        """
        al_model = self.model
        slack_model = self.model.model
        al_model.pi -= al_model.rho * convals
        if slack_model.m != 0:
            self.log.debug('New multipliers = %g, %g' %
                           (max(al_model.pi), min(al_model.pi)))

        if status == 'opt':
            # Safeguard: tighten tolerances only if desired optimality
            # is reached to prevent rapid decay of the tolerances from failed
            # inner loops
            self.eta /= al_model.rho**self.b_eta
            self.omega /= al_model.rho**self.b_omega
            self.inner_fail_count = 0
        else:
            self.inner_fail_count += 1
        return

    def update_penalty_parameter(self):
        """
        Large infeasibility; increase rho and reset tolerances
        based on new rho.
        """
        al_model = self.model
        al_model.rho /= self.tau
        self.eta = self.eta0 * al_model.rho**-self.a_eta
        self.omega = self.omega0 * al_model.rho**-self.a_omega
        return

    def post_iteration(self, **kwargs):
        """
        Override this method to perform additional work at the end of a
        major iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None

    def setup_bc_solver(self):
        """Setup bound-constrained solver."""
        return self.bc_solver(self.model, TruncatedCG, greltol=self.omega,
                              x0=self.x)

    def check_bc_solver_status(self, status):
        """Check if bound-constrained solver successfuly exited."""
        return "opt" if status in ['gtol'] else ""

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

        # Use a least-squares estimate of the multipliers to start (if
        # requested)
        if self.least_squares_pi and slack_model.m != 0:
            self.least_squares_multipliers(self.x)
            self.log.debug('New multipliers = %g, %g' %
                           (max(al_model.pi), min(al_model.pi)))

        # First augmented lagrangian gradient evaluation
        dphi = al_model.grad(self.x)

        # "Smart" initialization of slack variables using the magical step
        # function that is already available
        m_step_init = self.magical_step(self.x, dphi)
        self.x += m_step_init

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
        # Convergence check
        exitOptimal = (Pmax <= self.omega_opt and max_cons <= self.eta_opt)
        if exitOptimal:
            self.status = "opt"

        tick = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0:
            self.log.info(self.header)
            self.log.info(self.format0, self.iter, self.f, self.pg0,
                          self.cons0, al_model.rho, "", "", self.omega, self.eta)

        while not (exitOptimal or exitIter):
            self.iter += 1

            # Perform bound-constrained minimization
            bc_solver = self.setup_bc_solver()
            bc_solver.solve()
            self.x = bc_solver.x.copy()  # may not be useful.

            dL = al_model.dual_feasibility(self.x)
            PdL = self.project_gradient(self.x, dL)
            Pmax_new = np.max(np.abs(PdL))
            convals_new = slack_model.cons(self.x)

            # Specific handling for the case where the original NLP is
            # unconstrained
            if slack_model.m == 0:
                max_cons_new = 0.
            else:
                max_cons_new = np.max(np.abs(convals_new))

            self.f = self.model.model.model.obj(self.x[:on])
            self.pgnorm = Pmax_new

            # Print out header, say, every 20 iterations.
            if self.iter % 20 == 0:
                self.log.info(self.header)

            self.log.info(self.format % (self.iter, self.f,
                                         self.pgnorm, max_cons_new,
                                         al_model.rho,
                                         bc_solver.iter, bc_solver.status,
                                         self.omega, self.eta))

            # Update penalty parameter or multipliers based on result
            if max_cons_new <= np.maximum(self.eta, self.eta_opt):

                # Update convergence check
                if max_cons_new <= self.eta_opt and Pmax_new <= self.omega_opt:
                    exitOptimal = True
                    break

                bc_solver_status = self.check_bc_solver_status(bc_solver.status)
                self.update_multipliers(convals_new, bc_solver_status)

                # Update reference constraint norm on successful reduction
                cons_norm_ref = max_cons_new
                infeas_iter = 0

                # If optimality of the inner loop is not achieved within 10
                # major iterations, exit immediately
                if self.inner_fail_count == 10:
                    self.status = "fail"
                    self.log.debug(
                        'Current point could not be improved, exiting ... \n')
                    break

                self.log.debug(
                    '******  Updating multipliers estimates  ******\n')

            else:

                self.update_penalty_parameter()
                self.log.debug(
                    '******  Keeping current multipliers estimates  ******\n')

                if max_cons_new > 0.99 * cons_norm_ref and self.iter != 1:
                    infeas_iter += 1
                else:
                    cons_norm_ref = max_cons_new
                    infeas_iter = 0

                if infeas_iter == 10:
                    self.status = "stal"
                    self.log.debug('Problem appears to be infeasible,' +
                                   'exiting ... \n')
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

            exitIter = self.niter_total > self.max_inner_iter

        self.tsolve = cputime() - tick    # Solve time
        if slack_model.m != 0:
            self.pi_max = np.max(np.abs(al_model.pi))
            self.cons_max = np.max(np.abs(slack_model.cons(self.x)))
        else:
            self.pi_max = None
            self.cons_max = None

        # Solution output, etc.
        if exitOptimal:
            self.status = "opt"
            self.log.debug('Optimal solution found \n')
        elif not exitOptimal and self.status is None:
            self.status = "iter"
            self.log.debug('Maximum number of iterations reached \n')

        self.log.info('f = %12.8g' % self.f)
        if slack_model.m != 0:
            self.log.info('pi_max = %12.8g' % np.max(al_model.pi))
            self.log.info('max infeas. = %12.8g' % max_cons_new)
