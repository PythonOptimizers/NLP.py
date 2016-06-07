# -*- coding: utf-8 -*-

from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.model.linemodel import C1LineModel
from nlp.ls import ArmijoLineSearch

from nlp.tools.exceptions import UserExitRequest
from nlp.tools.norms import norm2
from nlp.tools.timing import cputime

try:
    from hsl.solvers.pyma57 import PyMa57Solver as LBLSolver
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLSolver

import pysparse.sparse.pysparseMatrix as ps

import numpy as np
import logging
import sys

np.set_printoptions(precision=16, formatter={'float': '{:0.8g}'.format})


class RegSQPSolver(object):
    """
    A regularized SQP method for degenerate equality-constrained optimization.
    """

    def __init__(self, model, **kwargs):
        u"""Regularized SQP framework for an equality-constrained problem.

        :keywords:
            :model: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for ‖[g-J'y ; c]‖
            :theta: sufficient decrease condition for the inner iterations
            :prox: initial proximal parameter
            :penalty: initial penalty parameter
            :itermax: maximum number of iterations allowed
            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

        """
        self.model = model
        self.x = model.x0.copy()
        self.y = np.ones(model.m)

        self.abstol = kwargs.get('abstol', 1.0e-7)
        self.reltol = kwargs.get('reltol', 1.0e-7)
        self.theta = kwargs.get('theta', 0.99)
        self.itermax = kwargs.get('maxiter', max(100, 10 * model.n))
        self.save_g = kwargs.get('save_g', False)

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 5)

        self.K = None
        self.LBL = None

        # Set regularization parameters.
        self.prox_min = 1.0e-8
        self.penalty_min = 1.0e-8
        prox = max(self.prox_min, kwargs.get('prox', 1.0))
        penalty = max(self.penalty_min, kwargs.get('penalty', 1.0))
        self.merit = AugmentedLagrangian(model,
                                         penalty=1./penalty,
                                         prox=prox,
                                         xk=model.x0.copy())

        # Initialize format strings for display
        self.hformat = "%-5s  %8s  %7s  %7s  %6s  %8s  %8s"
        self.header = self.hformat % (
            "iter", "f", u"‖c‖", u"‖∇L‖", "inner", u"ρ", u"δ")

        self.format = "%-5d  %8.1e  %7.1e  %7.1e  %6d  %8.1e  %8.1e"
        self.format0 = "%-5d  %8.1e  %7.1e  %7.1e  %6s  %8.1e  %8.1e"

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'nlp.regsqp')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False
        return

    def assemble_linear_system(self, x, y, delta, return_H=False, **kwargs):
        """Assemble main saddle-point matrix.

        [ H+ρI      J' ] [∆x] = [ -g + J'y ]
        [    J     -δI ] [∆y]   [ -c       ]

        For now H is the exact Hessian of the Lagrangian.
        """
        self.log.debug('assembling linear system')

        # Some shortcuts for convenience
        model = self.model
        n = model.n
        m = model.m
        self.K = ps.PysparseMatrix(nrow=n+m, ncol=n+m, symmetric=True)

        # contribution of the Hessian
        H = model.hess(x, z=y, **kwargs)
        diagH = H.takeDiagonal()
        (H_val, H_irow, H_jcol) = H.find()
        self.K.put(H_val, H_irow.tolist(), H_jcol.tolist())

        # add primal regularization
        self.K.addAt(self.rho * np.ones(n), range(n), range(n))

        # contribution of the Jacobian
        J = model.jac(x)
        (val, irow, jcol) = J.find()
        self.K.put(val, (n + irow).tolist(), jcol.tolist())  # dpo: tolist() necessary?

        # dual regularization
        self.K.put(-delta * np.ones(m), range(n, n + m), range(n, m + n))

        if return_H:
            return diagH, H
        else:
            return

    def initialize_rhs(self):
        return np.zeros(self.model.n + self.model.m)

    def update_rhs(self, rhs, g, J, y, c):
        n = self.model.n
        rhs[:n] = -g + J.T * y
        rhs[n:] = -c
        return

    def new_penalty(self, penalty, Fnorm):
        """Return updated penalty parameter value."""
        alpha = 0.9
        gamma = 1.1
        penalty = max(min(Fnorm, min(alpha * penalty, penalty**gamma)),
                      self.penalty_min)
        return penalty

    # def phi(self, x, y, rho, delta, x0, f=None, c=None):
    #     # dpo: this function should be in its own class as a merit function.
    #     model = self.model
    #     if f is None:
    #         f = model.obj(x)
    #     if c is None:
    #         c = model.cons(x) - model.Lcon
    #     phi = f - np.dot(c, y) + 0.5 / delta * \
    #         norm2(c)**2  # + 0.5 * rho * norm2(x - x0)**2
    #         # dpo: rho is needed here! What is x0? It should be the current
    #         # iterate. Is that self.x?
    #     return phi
    #
    # def dphi(self, x, y, rho, delta, x0, g=None, c=None, J=None,):
    #     model = self.model
    #     if g is None:
    #         g = model.grad(x)
    #     if c is None:
    #         c = model.cons(x) - model.Lcon
    #     if J is None:
    #         J = model.jop(x)
    #     dphi = g - J.T * (y - c / delta)  # + rho * (x - x0)
    #     # dpo: same comment about rho.
    #     return dphi

    # def backtracking_linesearch(self, x, y, dx, delta,
    #                             f=None, g=None, c=None, J=None,
    #                             bkmax=50, armijo=1.0e-2):
    #     """
    #     Perform a simple backtracking linesearch on the merit function
    #     from `x` along `step`.
    #
    #     Return (new_x, new_y, phi, steplength), where `new_x = x + steplength * step`
    #     satisfies the Armijo condition and `phi` is the merit function value at
    #     this new point.
    #     """
    #     self.log.debug('Entering backtracking linesearch')
    #
    #     x_trial = x + dx
    #     phi = self.phi(x, y, self.rho, delta, x, f, c)
    #     phi_trial = self.phi(x_trial, y, self.rho, delta, x_trial)
    #     g = self.dphi(x, y, self.rho, delta, x, g, c, J)
    #
    #     slope = np.dot(g, dx)
    #     self.log.debug('    slope: %6.2e', slope)
    #
    #     if slope >= 0:
    #         raise ValueError('ERROR: negative slope')
    #
    #     bk = 0
    #     alpha = 1.0
    #     while bk < bkmax and \
    #             phi_trial >= phi + armijo * alpha * slope:
    #         bk = bk + 1
    #         alpha /= 1.2
    #         x_trial = x + alpha * dx
    #         phi_trial = self.phi(x_trial, y, self.rho, delta, x_trial)
    #
    #     self.log.debug('    alpha=%3.2e, phi0=%3.2e, phi=%3.2e',
    #                    alpha, phi, phi_trial)
    #     self.log.debug('Leaving backtracking linesearch')
    #
    #     return (x_trial, phi_trial, alpha)

    def solve_linear_system(self, rhs, delta,
                            diagH=None, J=None,
                            itref_threshold=1.0e-10,
                            nitrefmax=5, **kwargs):

        self.log.debug('Solving linear system')
        nvar = self.model.nvar
        ncon = self.model.ncon
        prox_factor = 10
        penalty_factor = 10

        self.LBL = LBLSolver(self.K, factorize=True)
        second_order_sufficient = self.LBL.inertia == (nvar, ncon, 0)
        full_rank = self.LBL.isFullRank

        nb_bump = 0
        while not (second_order_sufficient and full_rank):
            if not second_order_sufficient:
                # further convexify
                self.K.addAt(prox_factor * self.merit.prox * np.ones(nvar),
                             range(nvar), range(nvar))
                self.merit.prox *= prox_factor + 1

            if not full_rank:
                # further regularize; this isn't quite supported by theory
                # the augmented Lagrangian uses 1/δ
                self.K.addAt(-penalty_factor / self.merit.penalty * np.ones(ncon),
                             range(nvar, nvar + ncon),
                             range(nvar, nvar + ncon))
                self.merit.penalty /= penalty_factor + 1

            self.LBL = LBLSolver(self.K, factorize=True)
            second_order_sufficient = self.LBL.inertia == (nvar, ncon, 0)
            full_rank = self.LBL.isFullRank
            nb_bump += 1

        if not self.LBL.isFullRank:
            if diagH is None:
                msg = 'A correction of inertia is needed, but diag_H is not '
                msg += 'provided.'
                raise ValueError(msg)
            while not factorized and not degenerate:
                self.log.debug('    A correction of inertia is needed.')
                self.K.put(diagH + self.rho * np.ones(n), range(n), range(n))
                self.LBL = LBLSolver(self.K, factorize=True, sqd=True)
                factorized = True

                # If the augmented matrix does not have full rank, bump up the
                # regularization parameters.
                if not self.LBL.isFullRank:
                    if self.rho > 0:
                        self.rho *= 100
                    else:
                        self.rho = self.rho_min
                    nb_bump += 1
                    degenerate = nb_bump > self.bump_max
                    factorized = False

        # Abandon if regularization is unsuccessful.
        if not self.LBL.isFullRank and degenerate:
            status = '    Unable to regularize sufficiently.'
            short_status = 'degn'
            finished = True
            dx = None
            dy = None
        else:
            self.LBL.solve(rhs)
            self.LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
            (dx, dy) = self.get_dx_dy(self.LBL.x)
            self.log.debug('    residual norm: %3.2e',
                           norm2(self.LBL.residual))
            status = None
            short_status = None
            finished = False

        self.log.debug('Leaving linear system')

        return status, short_status, finished, nb_bump, dx, dy

    def get_dx_dy(self, step):
        """
        Split `step` into steps along x and y.
        Outputs are *references*, not copies.
        """
        self.log.debug('Splitting step')
        n = self.model.n
        dx = step[:n]
        dy = -step[n:]
        return (dx, dy)

    def solve(self, **kwargs):

        # Transfer pointers for convenience.
        itermax = self.itermax
        delta = self.delta
        model = self.model
        x = self.x
        y = self.y
        theta = self.theta
        self.short_status = "fail"
        self.status = "fail"
        self.tsolve = 0

        ls_fmt = "%7.1e  %8.1e"

        # Get initial objective value
        print 'x0: ', x
        self.f0 = model.obj(x)
        self.x_old = x.copy()

        # Initialize right-hand side and coefficient matrix
        # of linear systems
        rhs = self.initialize_rhs()

        f = self.f = self.f0
        g = model.grad(x)
        J = model.jop(x)
        c = model.cons(x) - model.Lcon
        cnorm = self.cnorm = self.cnorm0 = norm2(c)
        grad_L = g - J.T * y
        grad_L_norm = self.grad_L_norm = self.grad_L_norm0 = norm2(grad_L)
        Fnorm = Fnorm0 = np.sqrt(grad_L_norm**2 + cnorm**2)

        # Find a better initial point
        self.assemble_linear_system(x, y, 0)
        self.update_rhs(rhs, g, J, y, c)

        status, short_status, finished, nbumps, dx, dy = self.solve_linear_system(
            rhs, delta, J=J)

        xs = x + dx
        ys = y + dy
        gs = model.grad(xs)
        Js = model.jop(xs)
        cs = model.cons(xs) - model.Lcon
        grad_Ls = gs - Js.T * ys
        Fnorms = np.sqrt(norm2(grad_Ls)**2 + norm2(cs)**2)
        if Fnorms <= Fnorm0:
            x += dx
            y += dy
            Fnorm = Fnorm0 = Fnorms
            g = gs.copy()
            J = model.jop(x)
            c = cs.copy()

        # Initialize penalty parameter
        delta = min(0.1, Fnorm0)

        tol = self.reltol * Fnorm0 + self.abstol

        optimal = (Fnorm0 <= tol)
        if optimal:
            status = 'Optimal solution found'
            short_status = 'opt'

        self.iter = 0
        tick = cputime()
        finished = False or optimal

        # Display initial header every so often.
        self.log.info(self.header)
        self.log.debug(norm2(g - J.T * y))
        self.log.info(self.format0, self.iter, self.f0, self.cnorm0,
                      self.grad_L_norm0, "", self.rho, delta)

        # Main loop.
        while not finished:
            self.x_old = x.copy()
            self.gL_old = grad_L.copy()

            # Step 2
            self.merit.penalty = 1 / self.new_penalty(delta, Fnorm)

            diagH, H = self.assemble_linear_system(x, y, delta, return_H=True)
            self.update_rhs(rhs, g, J, y, c)

            status, short_status, finished, nbumps, dx, dy = self.solve_linear_system(
                rhs, delta, diagH, J)

            # Step 3
            epsilon = 10 * delta  # a better way to set epsilon dynamically?

            # Step 4: Inner Iterations
            x_trial = x + dx
            y_trial = y + dy
            g_trial = model.grad(x_trial)
            J_trial = model.jop(x_trial)
            c_trial = model.cons(x_trial) - model.Lcon
            # Fnorm = norm_infty(g - J.T * y) + norm_infty(c)

            grad_L_trial = g_trial - J_trial.T * y_trial
            grad_L_trial_norm = norm2(grad_L_trial)
            c_trial_norm = norm2(c_trial)
            F_trialnorm = np.sqrt(grad_L_trial_norm**2 + c_trial_norm**2)

            inner_iter = 0

            if F_trialnorm > theta * Fnorm + epsilon:  # and inner_iter < 20:
                self.log.debug('Entering inner iterations loop')
                self.log.debug('    condition: %6.2e > %6.2e',
                               F_trialnorm, theta * Fnorm + epsilon)
                grad_L_trial_norm0 = grad_L_trial_norm
                c_trial_norm0 = c_trial_norm
                leave = False
                grad_phi0 = g_trial - J_trial.T * (y_trial - c_trial / delta)
                if norm2(grad_phi0) <= theta * grad_L_trial_norm0 + 0.5 * epsilon:
                    if c_trial_norm0 > theta * c_trial_norm0 + 0.5 * epsilon:
                        delta /= 10

                while not leave:
                    self.x_old = x_trial.copy()
                    self.gL_old = grad_L_trial.copy()

                    # Step 3: Compute a new direction p_j
                    diagH, H = self.assemble_linear_system(x_trial, y_trial,
                                                         delta, return_H=True)
                    self.update_rhs(rhs, g_trial, J_trial, y_trial, c_trial)

                    status, short_status, finished, nbumps, dx_trial, _ = self.solve_linear_system(
                        rhs, delta, diagH, J, inner=True)

                    # Break inner iteration loop if inertia correction fails
                    # if finished:
                    #     break

                    # Step 4: Armijo backtracking linesearch
                    self.merit.pi = y_trial
                    line_model = C1LineModel(self.merit, x_trial, dx_trial)
                    ls = ArmijoLineSearch(line_model, bkmax=5, decr=1.75)

                    try:
                        for step in ls:
                            self.log.debug(ls_fmt, step, ls.trial_value)

                    except LineSearchFailure:
                        step_status = "Rej"

                    (x_trial, phi_kj, alpha) = self.backtracking_linesearch(
                        x_trial, y_trial, dx_trial, delta)

                    g_trial = model.grad(x_trial)
                    J_trial = model.jop(x_trial)
                    c_trial = model.cons(x_trial) - model.Lcon
                    c_trial_norm = norm2(c_trial)
                    grad_L_trial = g_trial - J_trial.T * y_trial
                    inner_iter += 1
                    grad_phi_trial = g_trial - J_trial.T * \
                        (y_trial - c_trial / delta)

                    if norm2(grad_phi_trial) <= theta * grad_L_trial_norm0 + 0.5 * epsilon:
                        if c_trial_norm <= theta * c_trial_norm0 + 0.5 * epsilon:
                            self.log.debug('Leaving inner iterations loop')
                            leave = True
                            y_trial -= c_trial / delta
                        else:
                            delta /= 10

                    try:
                        self.post_inner_iteration(x_trial, grad_L_trial)
                    except UserExitRequest:
                        self.status = "User exit"

            # Update values of the new iterate and compute stopping criterion.
            x = x_trial.copy()
            y = y_trial.copy()
            g = g_trial.copy()
            J = model.jop(x_trial)
            c = c_trial.copy()
            cnorm = norm2(c)
            grad_L = g - J.T * y
            grad_L_norm = norm2(grad_L)
            self.log.debug(' condition outer: %6.2e <= %6.2e',
                           np.sqrt(grad_L_norm**2 + cnorm**2), theta * Fnorm + epsilon)
            Fnorm = np.sqrt(grad_L_norm**2 + cnorm**2)
            optimal = (Fnorm <= tol)

            if inner_iter == 0:
                try:
                    self.post_iteration(x, grad_L)
                except UserExitRequest:
                    self.status = "User exit"
                    short_status = 'user'

            # Display initial header every so often.
            # if iter % 50 == 49:
            if True:
                self.log.info(self.header)

            self.iter += inner_iter + 1
            f = model.obj(x)
            self.log.info(self.format, self.iter, f, cnorm,
                          grad_L_norm, inner_iter, self.rho, delta)

            if optimal:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if self.iter >= itermax:
                status = 'Maximum number of iterations reached'
                short_status = 'iter'
                finished = True
                continue

        # Transfer final values to class members.
        self.tsolve = cputime() - tick
        self.x = x.copy()
        self.y = y.copy()
        self.f = f
        self.cnorm = cnorm
        self.grad_L_norm = grad_L_norm
        self.optimal = optimal
        self.delta = delta
        self.status = status
        self.short_status = short_status
        return

    def post_iteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        major iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None

    def post_inner_iteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        minor iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None


if __name__ == '__main__':

    # Create root logger.
    log = logging.getLogger('regsqp')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    # Configure the solver logger.
    sublogger = logging.getLogger('regsqp.solver')
    sublogger.setLevel(logging.DEBUG)
    sublogger.addHandler(hndlr)
    sublogger.propagate = False

    filename = '/Users/syarra/work/programs/CuteExamples/nl_folder/hs007.nl'
    # test_RegSQPSolver(filename)

    model = PySparseAmplModel(filename)         # Create a model
    # model.x0 = np.array([0,np.søqrt(3)])
    solver = RegSQPSolver(model)
    solver.solve()
    print 'x:', solver.x
    print solver.status
