# -*- coding: utf-8 -*-
"""Factorization-free regularized SQP solver."""

# from pykrylov.linop.blkop import BlockLinearOperator
from pykrylov.linop import IdentityOperator  # , LinearOperator
# from pykrylov.lls import LSMRFramework
from reg_lsmr import RegLSMRFramework
# from nlp.tools.norms import norm2, norm_infty
# from nlp.tools.timing import cputime

# from pykrylov.linop import InverseLBFGSOperator
from damped_lbfgs import DampedInverseLBFGSOperator

# from nlp.tools.exceptions import UserExitRequest

from new_regsqp import RegSQPSolver


class RegSQPBFGSIterativeSolver(RegSQPSolver):
    """Factorization-free regularized SQP solver.

    A regularized SQP method for degenerate equality-constrained optimization.
    Using an iterative method to solve the system.
    """

    def __init__(self, model, **kwargs):
        """Instantiate a regularized SQP iterative.

        :keywords:
            :model: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for || [g-J'y ; c] ||
            :theta: sufficient decrease condition for the inner iterations
            :itermax: maximum number of iterations allowed
        """
        super(RegSQPBFGSIterativeSolver, self).__init__(model, **kwargs)

        # Create some DiagonalOperators and save them.
        self.Im = IdentityOperator(model.m)

        # no primal regularization when using LBFGS
        self.rho = 0

        # System solve method.
        self.iterative_solver = RegLSMRFramework

        # self.HinvOp = InverseLBFGSOperator(model.n,
        #                                    npairs=kwargs.get('qn_pairs', 6),
        #                                    scaling=True, **kwargs)
        self.HinvOp = DampedInverseLBFGSOperator(model.n,
                                                 npairs=kwargs.get('qn_pairs', 6),
                                                 scaling=True, **kwargs)

        return

    def least_squares_multipliers(self, g, x):
        u"""Compute least-squares multipliers at x.

        Compute least-squares multipliers y by solving

            min_y   1/2 ‖J(x)'y - g(x)‖²,

        which we do by solving
             [I    J(x)'] [r] = [g(x)]
             [J(x)    0 ] [y]   [ 0  ].

        We also return the residual r := g - J'y.
        """
        self.log.debug("computing least-squares multiplier estimates")
        J = self.model.jop(x)
        lsmr = self.iterative_solver(J.T)
        y, istop, itn, normr, normJr, normJ, condJ, normy = \
            lsmr.solve(g, show=False)
        self.log.debug("lsq: istop = %d, itn = %d", istop, itn)
        self.log.debug("lsq: ‖r‖ = %7.1e, ‖Jr‖ = %7.1e", normr, normJr)
        self.log.debug("lsq: ‖J‖ = %7.1e, cond(J) = %7.1e", normJ, condJ)

        r = g - J.T * y

        # update jprod counter
        self.jprod += J.nMatvec + J.T.nMatvec

        return r, y

    def assemble_linear_system(self, *args, **kwargs):
        """Assemble linear system.

        This method does nothing in the factorization-free version.
        """
        pass

    def shift_rhs(self, J, delta, rhs):
        """Shift rhs so it has the form (b, 0)."""
        n = self.model.n
        shifted_rhs = rhs[:n] + J.T * rhs[n:] / delta
        return shifted_rhs

    def get_dxdy(self, dy_tilde, J, shifted_rhs, delta, rhs):
        u"""Extract the (dx,dy) given dy_tilde.

        Given dy_tilde, the solution of the shifted linear system, extract
        dx and dy, the components of the solution of the original system.
        """
        n = self.model.n
        g = rhs[n:]  # g = -c_k
        dx = self.HinvOp * (shifted_rhs + J.T * dy_tilde)
        dy = dy_tilde.copy() + g / delta
        return dx, dy

    def solve_linear_system(self, rhs, x, J, inner=False,
                            itref_threshold=1.0e-15, nitrefmax=5, **kwargs):
        """Solve shifted linear system."""
        self.log.debug('Solving linear system')
        delta = 1.0 / self.merit.penalty
        J = self.model.jop(x)
        shifted_rhs = self.shift_rhs(J, delta, rhs)
        solver = self.iterative_solver(J.T)
        N = 1.0 / delta * self.Im
        dy_tilde, istop, itn, normr, normar, normA, condA, normx = \
            solver.solve(-shifted_rhs, M=self.HinvOp, N=N, damp=1.0,
                         atol=0.2 * min(1, self.merit.penalty**0.5),
                         etol=0.0, btol=1e-20,
                         inner=inner, show=False, **kwargs)

        dx, dy = self.get_dxdy(dy_tilde, J, shifted_rhs, delta, rhs)
        return '', '', istop in (0, 1, 2, 4, 5, 9, 8, 10), dx, dy

    def post_iteration(self, x, g, **kwargs):
        """Perform post-iteration work.

        This method updates the limited-memory quasi-Newton Hessian after
        every outer iteration.
        """
        s = x - self.x_old
        y = g - self.gL_old
        self.HinvOp.store(s, y)
        self.x_old = x.copy()
        self.gL_old = g.copy()
        return

    def post_inner_iteration(self, x, g, **kwargs):
        """Perform post-inner-iteration work.

        This method updates the limited-memory quasi-Newton Hessian after
        each inner iteration.
        """
        s = x - self.x_old
        y = g - self.gL_old
        self.HinvOp.store(s, y)
        self.x_old = x.copy()
        self.gL_old = g.copy()
