# -*- coding: utf-8 -*-

from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.model.nlpmodel import UnconstrainedNLPModel
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.model.linemodel import C1LineModel
from nlp.model.snlp import SlackModel

from nlp.ls.linesearch import LineSearch, ArmijoLineSearch
from nlp.ls.linesearch import ArmijoWolfeLineSearch, LineSearchFailure
from nlp.ls.wolfe import StrongWolfeLineSearch
from nlp.ls.quad_cub import QuadraticCubicLineSearch

from nlp.tools.exceptions import UserExitRequest
from nlp.tools.norms import norm2 as norm2
from nlp.tools.timing import cputime

try:
    from hsl.solvers.pyma57 import PyMa57Solver as LBLSolver
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLSolver

import pysparse.sparse.pysparseMatrix as ps
from pykrylov.linop import IdentityOperator

import logging
import math
import numpy as np


eps = np.finfo(np.double).eps
sqeps = math.sqrt(eps)
np.set_printoptions(precision=16, formatter={'float': '{:0.8g}'.format})
ls_fmt = "%7.1e  %8.2e"


class SimpleBacktrackingLineSearch(LineSearch):
    """Simple backtracking linesearch."""

    def __init__(self, *args, **kwargs):
        u"""Instantiate a simple backtracking linesearch.

        The search stops as soon as a step size t is found such that

            ϕ(t) <= Θ ϕ(0) + ϵ

        where 0 < Θ < 1 and ϵ > 0.

        :keywords:
            :decr: factor by which to reduce the steplength
                   during the backtracking (default: 1.5)
            :goal: Θ ϕ(0) + ϵ (default: √ϵM).
        """
        name = kwargs.pop("name", "Backtracking linesearch")
        super(SimpleBacktrackingLineSearch, self).__init__(
            *args, name=name, **kwargs)
        self.__goal = max(kwargs.get("goal", 0), sqeps)
        self.__decr = max(min(kwargs.get("decr", 1.5), 100), 1.001)
        return

    @property
    def decr(self):
        return self.__decr

    def check_slope(self, slope):
        # Don't check slope
        return

    def next(self):
        if self.trial_value <= self.__goal:
            raise StopIteration()

        step = self.step
        self._step /= self.decr
        if self.step < self.stepmin:
            raise LineSearchFailure("linesearch step too small")

        self._trial_iterate = self.linemodel.x + self.step * self.linemodel.d
        self._trial_value = self.linemodel.obj(self.step, x=self.iterate)

        return step  # return value of step just tested


class FnormModel(UnconstrainedNLPModel):

    def __init__(self, model, **kwargs):
        u"""Instantiate a :class:`FnormModel`.

        Objective function of this model is the norm of the KKT residual of
        `model` at w:=(x,y) defined by

            F(w) =  ‖∇L(w) + ρ(x-xₖ)‖ + ‖c(w) + δ(y-yₖ)‖

        :parameters:
            :model: ```NLPModel```.
        """
        if not isinstance(model, SlackModel):
            self.model = SlackModel(model, **kwargs)
        super(FnormModel, self).__init__(model.n + model.m, **kwargs)

        self.penalty_init = kwargs.get('penalty', 10.)
        self._penalty = self.penalty_init

        self.prox_init = kwargs.get("prox", 0.)
        self._prox = self.prox_init

        self.x0 = np.concatenate((self.model.x0, self.model.pi0))

        self.xk = kwargs.get("xk",
                             np.zeros(self.model.n) if self.prox_init > 0 else None)
        self.yk = kwargs.get("yk",
                             np.zeros(self.model.m) if self.penalty_init > 0 else None)

    @property
    def penalty(self):
        """Current penalty parameter."""
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        """Assign value to penalty parameter."""
        self._penalty = max(0, value)
        self.logger.debug("setting penalty parameter to %7.1e", self.penalty)

    @property
    def prox(self):
        """Current proximal parameter."""
        return self._prox

    @prox.setter
    def prox(self, value):
        """Assign value to proximal parameter."""
        self._prox = max(0, value)
        self.logger.debug("setting prox parameter to %7.1e", self.prox)

    def obj(self, w, c=None, g=None, gL=None):
        u"""Evaluate F(w)."""
        x = w[:self.model.n]
        y = w[self.model.n:]
        J = self.model.jop(x)

        if c is None:
            c = self.model.cons(x)
        if self.penalty > 0:
            c += self.penalty * (y - self.yk)
        cnorm = norm2(c)

        if gL is None:
            if g is None:
                g = self.model.grad(x)
            gL = g - J.T * y
        if self.prox > 0:
            gL += self.prox * (x - self.xk)
        gLnorm = norm2(gL)
        return gLnorm + cnorm

    def grad(self, w, c=None, g=None, gL=None):
        u"""Evaluate ∇ F(w)."""
        x = w[:self.model.n]
        y = w[self.model.n:]
        J = self.model.jop(x)
        H = self.model.hop(x, z=y)

        if c is None:
            c = self.model.cons(x)
        if self.penalty > 0:
            c += self.penalty * (y - self.yk)
        cnorm = norm2(c)
        if gL is None:
            if g is None:
                g = self.model.grad(x)
            gL = g - J.T * y
        if self.prox > 0:
            gL += self.prox * (x - self.xk)
        gLnorm = norm2(gL)

        if self.prox > 0:
            H = H + self.prox * IdentityOperator(self.model.n)
        dFdx = H * gL / gLnorm + J.T * c / cnorm
        dFdy = -(J * gL) / gLnorm
        if self.penalty > 0:
            dFdy += self.penalty * c / cnorm
        return np.concatenate((dFdx, dFdy))


class RegSQPSolver(object):
    """Regularized SQP method for equality-constrained optimization."""

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
        self.y = model.pi0.copy()

        self.abstol = kwargs.get('abstol', 1.0e-7)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.theta = kwargs.get('theta', 0.99)
        self.itermax = kwargs.get('maxiter', max(100, 10 * model.n))

        # attributed related to quasi-Newton variants
        self.save_g = kwargs.get('save_g', False)
        self.x_old = None
        self.gL_old = None

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 15)

        self.itn = 0
        self.K = None
        self.LBL = None
        self.short_status = "unknown"
        self.status = "unk"
        self.tsolve = 0.0
        self.f0 = None
        self.cnorm = None
        self.gLnorm = None

        # Set regularization parameters.
        # This solver uses the proximal augmented Lagrangian formulation
        # ϕ(x;yₖ,ρ,δ) := f(x) - c(x)ᵀyₖ + ½ ρ ‖x-xₖ‖² + ½ δ⁻¹ ‖c(x)‖².
        # Note the δ⁻¹.
        self.prox_min = 1.0e-3  # used when increasing the prox parameter
        self.prox_max = 1.0e40  # used in inertia
        self.prox_last = 0.0
        self.penalty_min = 1.0e-8
        prox = max(0.0, kwargs.get('prox', 0.0))
        penalty = max(self.penalty_min, kwargs.get('penalty', 10.0))
        self.merit = AugmentedLagrangian(model,
                                         penalty=1. / penalty,
                                         prox=prox,
                                         xk=model.x0.copy())
        # self.epsilon = (2 - self.theta) * penalty
        self.prox_factor = 10.  # increase factor during inertia correction
        self.penalty_factor = 10.  # increase factor during regularization

        # Initialize format strings for display
        self.hformat = "%-5s  %8s  %7s  %7s  %7s  %7s"
        self.header = self.hformat % ("iter", "f", u"‖c‖", u"‖∇L‖", u"ρ", u"δ")
        self.format = "%-5d  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e"

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'nlp.regsqp')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False
        return

    def assemble_linear_system(self, x, y, dual_reg=True):
        u"""Assemble main saddle-point matrix.

        [ H+ρI      Jᵀ ] [ ∆x] = [ -g + Jᵀy ]
        [    J     -δI ] [-∆y]   [ -c       ]

        For now H is the exact Hessian of the Lagrangian.
        """
        self.log.debug('assembling linear system')

        # Some shortcuts for convenience
        model = self.model
        n = model.n
        m = model.m
        self.K = ps.PysparseMatrix(nrow=n + m, ncol=n + m,
                                   sizeHint=model.nnzh + model.nnzj + m,
                                   symmetric=True)

        # contribution of the Hessian
        # print "x: ", x, " y: ", y
        f = model.obj(x)
        g = model.grad(x)
        J = model.jop(x)
        c = model.cons(x) - model.Lcon
        H = model.hess(x, z=y)
        (val, irow, jcol) = H.find()
        self.K.put(val, irow.tolist(), jcol.tolist())

        # add primal regularization
        # if self.merit.prox > 0:
        #     self.K.addAt(self.merit.prox * np.ones(n), range(n), range(n))

        # contribution of the Jacobian
        J = model.jac(x)
        (val, irow, jcol) = J.find()
        self.K.put(val, (n + irow).tolist(), jcol.tolist())

        # dual regularization
        if dual_reg:
            self.K.put(-1. / self.merit.penalty * np.ones(m),
                       range(n, n + m), range(n, m + n))
        return

    def assemble_rhs(self, g, J, y, c):
        """Set the rhs of Newton system according to current information."""
        # print g, J.to_array(), y, c
        return np.concatenate((-g + J.T * y, -c))

    def new_penalty(self, Fnorm):
        """Return updated penalty parameter value."""
        alpha = 0.9
        gamma = 1.1
        penalty = max(min(Fnorm,
                          min(alpha / self.merit.penalty,
                              1.0 / self.merit.penalty**gamma)),
                      self.penalty_min)
        # penalty = max(
        # min(1.0 / self.merit.penalty / 10., 1.0 / self.merit.penalty**(1 +
        #    0.8)), self.penalty_min)
        return penalty

    def solve_linear_system(self, rhs, itref_thresh=1.0e-7, nitref=5):
        u"""Compute a step by solving Newton's equations.

        Use a direct method to solve the symmetric and indefinite system

        [ H+ρI      J' ] [ ∆x] = [ -g + J'y ]
        [    J     -δI ] [-∆y]   [ -c       ].

        We increase ρ until the inertia indicates that H+ρI is positive
        definite on the nullspace of J and increase δ in case the matrix is
        singular because the rank deficiency in J.
        """
        self.log.debug('solving linear system, δ = %18.10e',
                       1. / self.merit.penalty)
        nvar = self.model.nvar
        ncon = self.model.ncon

        self.LBL = LBLSolver(self.K, factorize=True)

        inertia = self.LBL.inertia
        second_order_sufficient = inertia == (nvar, ncon, 0)
        full_rank = self.LBL.isFullRank

        self.merit.prox = 0.0

        nb_bump = 0
        tired = nb_bump > self.bump_max
        while not (second_order_sufficient and full_rank) and not tired:

            if not second_order_sufficient:
                self.log.debug("further convexifying model: inertia is " +
                               "(%d,%d,%d) and should be (%d,%d,%d))",
                               inertia[0], inertia[1], inertia[2],
                               self.model.nvar, self.model.ncon, 0)

                if self.merit.prox == 0.0:
                    if self.prox_last == 0.0:
                        self.merit.prox = self.prox_min
                    else:
                        self.merit.prox = max(self.prox_min,
                                              0.3 * self.prox_last)
                    self.K.addAt(self.merit.prox * np.ones(nvar),
                                 range(nvar), range(nvar))
                else:
                    if self.prox_last == 0.0:
                        factor = 100.
                    else:
                        factor = 8.
                    self.K.addAt(
                        factor * self.merit.prox * np.ones(nvar),
                        range(nvar), range(nvar))
                    self.merit.prox *= factor + 1

            if not full_rank:
                self.log.debug("further regularizing")
                # further regularize; this isn't quite supported by theory
                # the augmented Lagrangian uses 1/δ
                self.K.addAt(
                    -self.penalty_factor / self.merit.penalty * np.ones(ncon),
                    range(nvar, nvar + ncon), range(nvar, nvar + ncon))
                self.merit.penalty *= self.penalty_factor + 1

            self.LBL = LBLSolver(self.K, factorize=True)
            inertia = self.LBL.inertia
            second_order_sufficient = inertia == (nvar, ncon, 0)

            full_rank = self.LBL.isFullRank
            nb_bump += 1
            tired = nb_bump > self.bump_max

        if not second_order_sufficient:
            self.log.info("unable to convexify sufficiently")
            status = '    Unable to convexify sufficiently.'
            short_status = 'cnvx'
            solved = False
            dx = None
            dy = None
            return status, short_status, solved, dx, dy

        if not full_rank:
            self.log.info("unable to regularize sufficiently")
            status = '    Unable to regularize sufficiently.'
            short_status = 'degn'
            solved = False
            dx = None
            dy = None
            return status, short_status, solved, dx, dy

        self.prox_last = self.merit.prox
        self.LBL.solve(rhs)
        self.LBL.refine(rhs, nitref=nitref)
        (dx, dy) = self.get_dx_dy(self.LBL.x)
        self.log.debug("residual norm = %8.2e, ‖Δx‖ = %8.2e, ‖Δy‖ = %8.2e",
                       norm2(self.LBL.residual), norm2(dx), norm2(dy))

        # only for debugging, should be removed later
        J = self.K[nvar:, :nvar]
        HprIpdjj = np.dot(self.K[:nvar, :nvar] * dx, dx)
        HprIpdjj += self.merit.penalty * np.dot(dx, (J * dx) * J)
        if HprIpdjj <= 0:
            self.log.error("not sufficiently positive definite: %6.2e>=%6.2e",
                           HprIpdjj,
                           1e-8 * np.dot(dx, dx))
            assert HprIpdjj > 0

        status = None
        short_status = None
        solved = True
        return status, short_status, solved, dx, dy

    def get_dx_dy(self, step):
        """Split `step` into steps along x and y.

        Outputs are *references*, not copies.
        """
        return (step[:self.model.n], -step[self.model.n:])

    def solve_inner(self, x, y, f, g, J, c, gL,
                    Fnorm0, gLnorm0, cnorm0, Fnorm, gLnorm, cnorm):
        u"""Perform a sequence of inner iterations.

        The objective of the inner iterations is to identify an improved
        iterate w+ = (x+, y+) such that the optimality residual satisfies
        ‖F(w+)‖ ≤ Θ ‖F(w)‖ + ϵ.
        The improved iterate is identified by minimizing the proximal augmented
        Lagrangian.
        """
        # self.merit.penalty = 1. / Fnorm
        self.log.debug('starting inner iterations with target %7.1e',
                       self.theta * Fnorm0 + self.epsilon)
        self.log.info(self.format, self.itn, f, cnorm, gLnorm,
                      self.merit.prox, 1.0 / self.merit.penalty)

        model = self.model
        failure = False
        # ipdb.set_trace()

        # if infeasibility is large, immediately increase penalty
        # if cnorm > self.theta * cnorm0 + 0.5 * self.epsilon:
        #     self.merit.penalty *= 10

        y_al = y - c * self.merit.penalty
        phi = f - np.dot(y, c) + 0.5 * self.merit.penalty * cnorm**2
        gphi = g - J.T * y_al
        gphi_norm = norm2(gphi)
        self.log.debug(u"‖∇L‖ = %6.2e, ‖∇ϕ‖ = %6.2e",
                       norm2(g - J.T * y), gphi_norm)
        if gphi_norm <= self.theta * gLnorm0 + 0.5 * self.epsilon:
            self.log.debug('optimality improved sufficiently')
            if cnorm <= self.theta * cnorm0 + 0.5 * self.epsilon:
                self.log.debug('feasibility improved sufficiently')
                y = y_al

        Fnorm = gphi_norm + cnorm
        finished = Fnorm <= self.theta * Fnorm0 + self.epsilon
        tired = self.itn > self.itermax

        while not (failure or finished or tired):

            self.x_old = x.copy()
            self.gL_old = gL.copy()

            self.merit.xk = x.copy()

            # compute step
            self.assemble_linear_system(x, y)
            rhs = self.assemble_rhs(g, J, y, c)

            status, short_status, solved, dx, _ = self.solve_linear_system(rhs)
            # print x, y, dx
            assert solved

            if not solved:
                failure = True
                continue

            # Step 4: Armijo backtracking linesearch
            line_model = C1LineModel(self.merit, x, dx)
            slope = np.dot(gphi, dx)

            self.log.debug(u"ϕ(x) = %9.2e, ∇ϕᵀΔx = %9.2e", phi, slope)
            ls = ArmijoLineSearch(line_model, bkmax=50,
                                  decr=1.75, value=phi, slope=slope)
            # ls = ArmijoWolfeLineSearch(line_model, step=1.0, bkmax=10,
            #                            decr=1.75, value=phi, slope=slope)
            # ls = StrongWolfeLineSearch(
            #     line_model, value=phi, slope=slope, gtol=0.1)
            # ls = QuadraticCubicLineSearch(line_model, bkmax=50,
            #                               value=phi, slope=slope)

            try:
                for k, step in enumerate(ls):
                    self.log.debug(ls_fmt, step, ls.trial_value)

                self.log.debug('step norm: %8.2e', norm2(x - ls.iterate))
                x = ls.iterate
                f = model.obj(x)
                g = model.grad(x)
                J = model.jop(x)
                c = model.cons(x) - model.Lcon
                cnorm = norm2(c)
                gL = g - J.T * y
                y_al = y - c * self.merit.penalty
                phi = f - np.dot(y, c) + 0.5 * self.merit.penalty * cnorm**2
                gphi = g - J.T * y_al
                gphi_norm = norm2(gphi)

                if gphi_norm <= self.theta * gLnorm0 + 0.5 * self.epsilon:
                    self.log.debug('optimality improved sufficiently')
                    if cnorm <= self.theta * cnorm0 + 0.5 * self.epsilon:
                        self.log.debug('feasibility improved sufficiently')
                        y = y_al
                    else:
                        self.merit.penalty *= 10

                self.itn += 1
                self.inner_itn += 1
                Fnorm = gphi_norm + cnorm
                finished = Fnorm <= self.theta * Fnorm0 + self.epsilon
                tired = self.itn > self.itermax

                self.log.info(self.format, self.itn, f, cnorm, gphi_norm,
                              self.merit.prox, 1.0 / self.merit.penalty)

                try:
                    self.post_inner_iteration(x, gL)

                except UserExitRequest:
                    self.status = "User exit"
                    finished = True

            except LineSearchFailure:
                self.status = "Linesearch failure"
                failure = True
            # plt.scatter(x_p, y_p)
            # ipdb.set_trace()
        solved = Fnorm <= self.theta * Fnorm0 + self.epsilon
        return x, y, f, g, J, c, gphi, gphi_norm, cnorm, Fnorm, solved

    def least_squares_multipliers(self, g, J):
        u"""Compute least-squares multipliers.

            min_y   ½ ‖Jᵀy - g‖²

        That is solving
             [I  Jᵀ] [ r] = [-g]
             [J  0 ] [-y]   [ 0]
        and keeping only y.
        """
        model = self.model
        n = model.nvar
        m = model.ncon
        M = ps.PysparseMatrix(nrow=n + m, ncol=n + m,
                              sizeHint=model.nnzj + n,
                              symmetric=True)

        M.put(np.ones(n), range(n), range(n))

        # contribution of the Jacobian
        (val, irow, jcol) = J.find()
        M.put(val, (n + irow).tolist(), jcol.tolist())

        rhs = np.zeros(n + m)
        rhs[:n] = g
        rhs[:n] *= -1.0

        LBL = LBLSolver(M, factorize=True)

        LBL.solve(rhs)
        LBL.refine(rhs, nitref=3)
        y = LBL.x[n:]
        y *= -1.0
        return y

    def solve(self, **kwargs):

        # Transfer pointers for convenience.
        model = self.model
        x = self.x
        y = self.y
        self.short_status = "fail"
        self.status = "fail"
        self.tsolve = 0.0
        self.inner_itn = 0

        # Get initial objective value
        # print 'x0: ', x
        self.f = self.f0 = f = model.obj(x)

        c = model.cons(x) - model.Lcon
        self.cnorm = cnorm = cnorm0 = norm2(c)

        g = model.grad(x)
        J = model.jop(x)
        gL = g - J.T * y

        # set Lagrange mutlipliers to least-squares estimates
        y = self.least_squares_multipliers(g, model.jac(x))

        J = model.jop(x)
        # print J.to_array()
        gL = g - J.T * y
        self.gLnorm = gLnorm = gLnorm0 = norm2(gL)
        Fnorm = Fnorm0 = gLnorm + cnorm

        self.log.info(self.header)
        self.log.info(self.format, self.itn, self.f0, cnorm0, gLnorm0,
                      self.merit.prox, 1.0 / self.merit.penalty)

        # Find a better initial point
        # x, y, f, c, g, J, cnorm, gLnorm, Fnorm = \
        #     self.find_better_starting_point(x, y, f, c, g, J,
        #                                     cnorm0, gLnorm0, Fnorm0)

        # Initialize penalty parameter
        self.merit.penalty = 1.0 / min(0.1, Fnorm)
        # self.merit.penalty = 1. / Fnorm
        # delta = min(0.1, Fnorm0)

        # set stopping tolerance
        tol = self.reltol * Fnorm0 + self.abstol

        self.tsolve = 0

        self.optimal = optimal = Fnorm <= tol
        if optimal:
            status = 'Optimal solution found'
            short_status = 'opt'

        tired = self.itn > self.itermax
        finished = optimal or tired

        self.itn = 0
        tick = cputime()

        # Main loop.
        while not finished:

            self.x_old = x.copy()
            self.gL_old = gL.copy()

            # update penalty parameter
            self.merit.penalty = 1.0 / self.new_penalty(Fnorm)

            # compute extrapolation step
            self.assemble_linear_system(x, y)
            # self.update_rhs(rhs, g, J, y, c)
            rhs = self.assemble_rhs(g, J, y, c)

            status, short_status, solved, dx, dy = self.solve_linear_system(
                rhs, J)
            # print 'dx_ext: ', dx
            assert solved
            penalty_ext = self.merit.penalty
            prox_ext = self.merit.prox
            self.log.debug("step norm: %6.2e", norm2(dx))

            # check for acceptance of extrapolation step
            # self.epsilon = (2 - self.theta) * Fnorm  # 10./ self.merit.penalty
            self.epsilon = 10. / self.merit.penalty
            # x += dx
            # y += dy
            xplus = x + dx
            yplus = y + dy
            f_ext = model.obj(xplus)  # only necessary for printing
            g_ext = model.grad(xplus)
            J_ext = model.jop(xplus)
            c_ext = model.cons(xplus) - model.Lcon

            gL_ext = g_ext - J_ext.T * yplus        # = ∇L(wk+)
            gLnorm_ext = norm2(gL_ext)
            cnorm_ext = norm2(c_ext)                # = ‖c(xk+)‖
            Fnorm_ext = gLnorm_ext + cnorm_ext      # = ‖F(wk+)‖

            if Fnorm_ext <= self.theta * Fnorm + self.epsilon:
                self.log.debug("extrapolation step accepted")
                x = xplus
                y = yplus
                f = f_ext
                g = g_ext
                J = J_ext
                c = c_ext
                gL = gL_ext
                gLnorm = gLnorm_ext
                cnorm = cnorm_ext
                Fnorm = Fnorm_ext

                try:
                    self.post_iteration(x, gL)
                except UserExitRequest:
                    self.status = "User exit"
                    short_status = 'user'

                self.itn += 1

            else:
                # perform a sequence of inner iterations
                # starting from the old x and the old y
                self.merit.pi = y.copy()

                (x_in, y_in, f_in, g_in, J_in, c_in, gL_in, gLnorm_in,
                 cnorm_in, Fnorm_in, solved) = self.solve_inner(x, y, f, g, J, c, gL,
                                                                Fnorm, gLnorm, cnorm,
                                                                Fnorm, gLnorm, cnorm)
                # x = x_in
                # y = y_in
                # f = f_in
                # g = g_in
                # J = J_in
                # c = c_in
                # gL = gL_in
                # gLnorm = gLnorm_in
                # cnorm = cnorm_in
                # Fnorm = Fnorm_in

                if not solved:
                    self.merit.penalty = penalty_ext
                    self.merit.prox = prox_ext
                    self.epsilon = (2 - self.theta) * Fnorm
                    x, y, f, g, J, c, gL, gLnorm, cnorm, Fnorm = \
                        self.emergency_backtrack(x, y, dx, dy, prox_ext,
                                                 penalty_ext,
                                                 Fnorm, Fnorm_ext)
                else:
                    x = x_in
                    y = y_in
                    f = f_in
                    g = g_in
                    J = J_in
                    c = c_in
                    gL = gL_in
                    gLnorm = gLnorm_in
                    cnorm = cnorm_in
                    Fnorm = Fnorm_in

            optimal = Fnorm <= tol
            tired = self.itn > self.itermax
            finished = optimal or tired
            if self.itn % 20 == 0:
                self.log.info(self.header)

            self.log.info(self.format, self.itn, f, cnorm, gLnorm,
                          self.merit.prox, 1.0 / self.merit.penalty)

            if optimal:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if tired:
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
        self.gLnorm = gLnorm
        self.optimal = optimal
        self.status = status
        self.short_status = short_status
        return

    def emergency_backtrack(self, x, y, dx, dy, prox, penalty,
                            Fnorm0, Fnorm_ext):
        """Backtrack on ‖F(w)‖."""
        model = self.model
        goal = self.theta * Fnorm0 + self.epsilon

        self.log.warning("Starting emergency backtrack with goal: %6.2e",
                         goal)

        fnorm_model = FnormModel(model, prox=0, penalty=0)
        w = np.concatenate((x, y))
        d = np.concatenate((dx, dy))

        line_model = C1LineModel(fnorm_model, w, d)
        ls = SimpleBacktrackingLineSearch(line_model, decr=1.75,
                                          value=Fnorm0, trial_value=Fnorm_ext,
                                          goal=goal)

        try:
            for step in ls:
                self.log.debug(ls_fmt, step, ls.trial_value)

            self.log.debug('step norm: %6.2e',
                           norm2(w - ls.iterate))
            w = ls.iterate
            x = w[:model.n]
            y = w[model.n:]
            f = model.obj(x)
            g = model.grad(x)
            J = model.jop(x)
            c = model.cons(x) - model.Lcon
            cnorm = norm2(c)
            gL = g - J.T * y
            gLnorm = norm2(gL)
            Fnorm = gLnorm + cnorm

            self.log.info(self.format, self.itn, f, cnorm, gLnorm,
                          self.merit.prox, 1.0 / self.merit.penalty)

            try:
                self.post_inner_iteration(x, gL)

            except UserExitRequest:
                self.status = "User exit"

        except LineSearchFailure:
            self.status = "Linesearch failure"

        return x, y, f, g, J, c, gL, gLnorm, cnorm, Fnorm

    def find_better_starting_point(self, x, y, f, c, g, J, cnorm,
                                   gLnorm, Fnorm):
        """Attempt to find a better starting point.

        TODO: description
        """
        model = self.model
        self.assemble_linear_system(x, y, dual_reg=False)
        rhs = self.assemble_rhs(g, J, y, c)

        self.log.debug("solving linear system")
        LBL = LBLSolver(self.K, factorize=True)
        LBL.solve(rhs)
        LBL.refine(rhs, nitref=3)
        (dx, dy) = self.get_dx_dy(LBL.x)
        self.log.debug("residual norm = %8.2e, ‖Δx‖ = %8.2e, ‖Δy‖ = %8.2e",
                       norm2(LBL.residual),  norm2(dx), norm2(dy))

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

            self.log.info(self.format, self.itn,
                          self.f, self.cnorm, self.gLnorm,
                          self.merit.prox, 1.0 / self.merit.penalty)
        return x, y, f, c, g, J, cnorm, gLnorm, Fnorm

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