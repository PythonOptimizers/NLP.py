# -*- coding: utf-8 -*-
"""Krylov version of RegSQP."""
from pykrylov.linop.blkop import BlockLinearOperator
from pykrylov.linop import IdentityOperator, LinearOperator
from pykrylov.lls import LSMRFramework
from reg_lsmr import RegLSMRFramework
from nlp.tools.norms import norm2, norm_infty
from nlp.tools.timing import cputime

from pykrylov.linop import LBFGSOperator, InverseLBFGSOperator
from damped_lbfgs import DampedLBFGSOperator, DampedInverseLBFGSOperator

from nlp.tools.exceptions import UserExitRequest

import numpy as np
import logging
import sys

from new_regsqp import RegSQPSolver


class RegSQPBFGSIterativeSolver(RegSQPSolver):
    """A regularized SQP method for equality-constrained optimization.

    Using an iterative method to solve the system.
    """

    def __init__(self, model, **kwargs):
        u"""Instantiate a regularized SQP iterative framework.

        :keywords:
            :model: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for ‖[g-Jᵀy ; c]‖
            :theta: sufficient decrease condition for the inner iterations
            :itermax: maximum number of iterations allowed
        """
        super(RegSQPBFGSIterativeSolver, self).__init__(model, **kwargs)

        # Create some DiagonalOperators and save them.
        self.Im = IdentityOperator(model.m)

        self.rho = 0

        # Linear system solver
        self.iterative_solver = RegLSMRFramework

        self.HinvOp = InverseLBFGSOperator(model.n, npairs=kwargs.get(
            'qn_pairs', 6), scaling=True, **kwargs)

        return

    def least_squares_multipliers(self, x, g):
        u"""Compute least-squares multipliers using a direct method.

            min_y   ½ ‖J(x)ᵀy - g‖²
        """
        J = self.model.jop(x)
        solver = LSMRFramework(J.T)
        y, istop, itn, normr, normar, normA, condA, normx = \
            solver.solve(g, show=True)
        return y

    def shift_rhs(self, J, rhs):
        u"""Shift right hand side of the linear system.

        [ -g + Jᵀy ] ⟹ [ -g + Jᵀ(y - c/δ) ]
        [ -c       ]
        """
        n = self.model.n
        shifted_rhs = J.T * rhs[n:]
        shifted_rhs *= self.merit.penalty
        shifted_rhs += rhs[:n]
        return shifted_rhs

    def get_dx_dy(self, dy_tilde, J, shifted_rhs, rhs):
        u"""Retrieve (Δx, Δy) from Δȳ.

        Δx = H^{-1} (JᵀΔȳ + b)
        Δy = Δȳ - c/δ
        """
        n = self.model.n
        g = rhs[n:]  # g = -cₖ
        dx = self.HinvOp * (shifted_rhs + J.T * dy_tilde)
        dy = dy_tilde.copy() + g * self.merit.penalty
        return dx, dy

    def solve_linear_system(self, rhs, J=None, itref_thresh=1.0e-15, nitref=5):
        u"""Compute a step by solving Newton's equations.

        Use a Krylov method to solve the symmetric and indefinite system

        [ H    Jᵀ ] [ ∆x] = [ -g + Jᵀy ]
        [ J   -δI ] [-∆y]   [ -c       ]

        ⟺  [ H    Jᵀ ] [ ∆x] = [ -g + Jᵀ(y - c/δ) ]
            [ J   -δI ] [-∆ȳ]   [ 0                ].

        ⟺  min_Δȳ  ½ ‖JᵀΔȳ + b‖²_M + ½ ‖Δȳ‖²_δ
            where M := H^{-1}
        """
        self.log.debug('solving linear system, δ = %18.10e',
                       1. / self.merit.penalty)

        shifted_rhs = self.shift_rhs(J, rhs)

        self.log.debug('Solving linear system')
        solver = self.iterative_solver(J.T)
        N = self.merit.penalty * self.Im
        dy_tilde, istop, itn, normr, normar, normA, condA, normx = \
            solver.solve(-shifted_rhs, M=self.HinvOp, N=N, damp=1.0,
                         atol=itref_thresh, etol=itref_thresh,
                         btol=1e-20, itnlim=1000000, show=True)

        dx, dy = self.get_dx_dy(dy_tilde, J, shifted_rhs, rhs)

        return istop, '', True, dx, dy

    def find_better_starting_point(self, x, y, f, c, g, J, cnorm,
                                   gLnorm, Fnorm):
        """Attempt to find a better starting point.

        TODO: description
        """
        model = self.model

        self.x_old = x.copy()
        self.gL_old = g - J.T * y

        rhs = self.assemble_rhs(g, J, y, c)

        shifted_rhs = self.shift_rhs(J, rhs)

        self.log.debug('Solving linear system')
        solver = self.iterative_solver(J.T)
        N = 1e-8 * self.Im
        dy_tilde, istop, itn, normr, normar, normA, condA, normx = \
            solver.solve(-shifted_rhs, M=self.HinvOp, N=N, damp=1.0,
                         itnlim=1000000, show=True)

        dx, dy = self.get_dx_dy(dy_tilde, J, shifted_rhs, rhs)

        xs = x + dx
        ys = y + dy
        gs = model.grad(xs)
        Js = model.jop(xs)
        cs = model.cons(xs) - model.Lcon
        gLs = gs - Js.T * ys
        Fnorms = norm2(gLs) + norm2(cs)
        if Fnorms < Fnorm:
            self.log.debug("improved initial point accepted")
            x += dx
            y += dy
            Fnorm = Fnorms
            g = gs.copy()
            J = model.jop(x)
            c = cs.copy()
            self.f = f = model.obj(x)
            self.cnorm = cnorm = norm2(c)
            self.gLnorm = gLnorm = norm2(gLs)

            try:
                self.post_iteration(x, gLs)
            except UserExitRequest:
                self.status = "User exit"

            self.log.info(self.format, self.itn,
                          self.f, self.cnorm, self.gLnorm,
                          self.merit.prox, 1.0 / self.merit.penalty,
                          0)  # ϵ
        return x, y, f, c, g, J, cnorm, gLnorm, Fnorm

    def post_iteration(self, x, g, **kwargs):
        """Perform some task after each outer iteration.

        This method resets the limited-memory quasi-Newton Hessian after
        every outer iteration.
        """
        s = x - self.x_old
        y = g - self.gL_old
        self.HinvOp.store(s, y)
        self.x_old = x.copy()
        self.gL_old = g.copy()
        return

    def post_inner_iteration(self, x, g, **kwargs):
        """Perform some task after each inner iteration.

        This method updates the limited-memory quasi-Newton Hessian by
        appending the most recent (s,y) pair to it and possibly discarding the
        oldest one if all the memory has been used.
        """
        s = x - self.x_old
        y = g - self.gL_old
        self.HinvOp.store(s, y)
        self.x_old = x.copy()
        self.gL_old = g.copy()

# if __name__ == '__main__':
#
#     from nlp.model.pysparsemodel import PySparseAmplModel
#
#     # Create root logger.
#     log = logging.getLogger('nlp')
#     log.setLevel(logging.INFO)
#     fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
#     hndlr = logging.StreamHandler(sys.stdout)
#     hndlr.setFormatter(fmt)
#     log.addHandler(hndlr)
#
#     # Configure the solver logger.
#     sublogger = logging.getLogger('nlp.regsqp')
#     sublogger.setLevel(logging.DEBUG)
#     sublogger.addHandler(hndlr)
#     sublogger.propagate = False
#
#     filename = '/Users/syarra/work/programs/CuteExamples/nl_folder/hs009.nl'
#
#     model = PySparseAmplModel(filename)         # Create a model
#     solver = RegSQPBFGSIterativeSolver(model)
#     solver.solve()
#     print 'x:', solver.x
#     print solver.status
