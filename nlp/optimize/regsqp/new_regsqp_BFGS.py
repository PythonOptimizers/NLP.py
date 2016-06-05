# -*- coding: utf-8 -*-
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
    """
    A regularized SQP method for degenerate equality-constrained optimization.
    Using an iterative method to solve the system.
    """

    def __init__(self, model, **kwargs):
        """
        Instantiate a regularized SQP iterative framework for a given equality-constrained
        problem.

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

        self.rho = 0

        # System solve method.
        self.iterative_outer_solver = LSMRFramework
        self.iterative_inner_solver = RegLSMRFramework

        self.HinvOp = InverseLBFGSOperator(model.n, npairs=kwargs.get(
            'qn_pairs', 6), scaling=True, **kwargs)

        return

    def update_linear_system(self, x, y, delta, return_H=False, **kwargs):
        # [ H+ρI    J' ] [∆x] = [ -g + J'y ]
        # [    J   -δI ] [∆y]   [ -c       ]
        #
        # H is BFGS approximation of the Hessian of the Lagrangian.

        if return_H:
            return None, None
        else:
            return

    def shift_rhs(self, J, delta, rhs):
        n = self.model.n
        shifted_rhs = rhs[:n] + J.T * rhs[n:] / delta
        return shifted_rhs

    def get_dxdy(self, dy_tilde, J, shifted_rhs, delta, rhs):
        n = self.model.n
        g = rhs[n:]  # g = -c_k
        dx = self.HinvOp * (shifted_rhs + J.T * dy_tilde)
        dy = dy_tilde.copy() + g / delta
        return dx, dy

    def solve_linear_system(self, rhs, delta, diagH=None, J=None, itref_threshold=1.0e-15, nitrefmax=5, **kwargs):

        shifted_rhs = self.shift_rhs(J, delta, rhs)

        self.log.debug('Solving linear system')
        solver = self.iterative_inner_solver(J.T)
        N = self.Im / delta
        dy_tilde, istop, itn, normr, normar, normA, condA, normx = solver.solve(-shifted_rhs, M=self.HinvOp, N=N, damp=1.0,
                                                                                atol=itref_threshold, etol=itref_threshold,
                                                                                btol=1e-20, itnlim=1000000, show=True, **kwargs)

        dx, dy = self.get_dxdy(dy_tilde, J, shifted_rhs, delta, rhs)
        return istop, '', '', 0, dx, dy

    def post_iteration(self, x, g, **kwargs):
        """
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
        """
        This method updates the limited-memory quasi-Newton Hessian by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        s = x - self.x_old
        y = g - self.gL_old
        self.HinvOp.store(s, y)
        self.x_old = x.copy()
        self.gL_old = g.copy()

if __name__ == '__main__':

    from nlp.model.pysparsemodel import PySparseAmplModel

    # Create root logger.
    log = logging.getLogger('nlp')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    # Configure the solver logger.
    sublogger = logging.getLogger('nlp.regsqp')
    sublogger.setLevel(logging.DEBUG)
    sublogger.addHandler(hndlr)
    sublogger.propagate = False

    filename = '/Users/syarra/work/programs/CuteExamples/nl_folder/hs009.nl'

    model = PySparseAmplModel(filename)         # Create a model
    solver = RegSQPBFGSIterativeSolver(model)
    solver.solve()
    print 'x:', solver.x
    print solver.status
