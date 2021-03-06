# -*- coding: utf-8 -*-
"""Long-step primal-dual interior-point method for convex QP.

From Algorithm IPF on p.110 of Stephen J. Wright's book
"Primal-Dual Interior-Point Methods", SIAM ed., 1997.
The method uses the augmented system formulation. These systems are solved
using MA27 or MA57.

D. Orban, Montreal 2009-2011.
"""
try:                            # To solve augmented systems
    from hsl.solvers.pyma57 import PyMa57Solver as LBLContext
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLContext
from hsl.scaling.mc29 import mc29ad
from pykrylov.linop import PysparseLinearOperator
from nlp.tools.norms import norm2, norm_infty, normest
from nlp.tools.timing import cputime
import logging

# for slack model
from nlp.model.nlpmodel import NLPModel
from nlp.model.qnmodel import QuasiNewtonModel
from pysparse.sparse.pysparseMatrix import PysparseMatrix
import numpy as np


# CQP needs equality constraints and bounds on variables as s>=0
class SlackModel(NLPModel):
    """Framework for converting an optimization problem to a form using slacks.

    In the latter problem, the only inequality constraints are bounds on
    the slack variables. The other constraints are (typically) nonlinear
    equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL = [ sLL | sLR ], sLL being the slack variables corresponding to
       general constraints with a lower bound only, and sLR being the slack
       variables corresponding to the 'lower' side of range constraints.

    3. sU = [ sUU | sUR ], sUU being the slack variables corresponding to
       general constraints with an upper bound only, and sUR being the slack
       variables corresponding to the 'upper' side of range constraints.

    4. tL = [ tLL | tLR ], tLL being the slack variables corresponding to
       variables with a lower bound only, and tLR being the slack variables
       corresponding to the 'lower' side of two-sided bounds.

    5. tU = [ tUU | tUR ], tUU being the slack variables corresponding to
       variables with an upper bound only, and tLR being the slack variables
       corresponding to the 'upper' side of two-sided bounds.

    This framework initializes the slack variables sL, sU, tL, and tU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, model, keep_variable_bounds=False, **kwargs):
        """Instantiate a model in its slack form.

        :parameters:
            :model:  Original NLP to transform to a slack form.

        :keywords:
            :keep_variable_bounds: set to `True` if you don't want to convert
                                   bounds on variables to inequalities. In this
                                   case bounds are kept as they were in the
                                   original NLP.
        """
        self.model = model
        self.keep_variable_bounds = keep_variable_bounds

        n_con_low = model.nlowerC + model.nrangeC  # ineqs with lower bound.
        n_con_upp = model.nupperC + model.nrangeC  # ineqs with upper bound.
        n_var_low = model.nlowerB + model.nrangeB  # vars  with lower bound.
        n_var_upp = model.nupperB + model.nrangeB  # vars  with upper bound.

        bot = model.n
        self.sLL = range(bot, bot + model.nlowerC)
        bot += model.nlowerC
        self.sLR = range(bot, bot + model.nrangeC)
        bot += model.nrangeC
        self.sUU = range(bot, bot + model.nupperC)
        bot += model.nupperC
        self.sUR = range(bot, bot + model.nrangeC)

        # Update effective number of variables and constraints
        if keep_variable_bounds:
            n = model.n + n_con_low + n_con_upp
            m = model.m + model.nrangeC

            Lvar = np.zeros(n)
            Lvar[:model.n] = model.Lvar
            Uvar = np.inf * np.ones(n)
            Uvar[:model.n] = model.Uvar

        else:
            bot += model.nrangeC
            self.tLL = range(bot, bot + model.nlowerB)
            bot += model.nlowerB
            self.tLR = range(bot, bot + model.nrangeB)
            bot += model.nrangeB
            self.tUU = range(bot, bot + model.nupperB)
            bot += model.nupperB
            self.tUR = range(bot, bot + model.nrangeB)

            n = model.n + n_con_low + n_con_upp + n_var_low + n_var_upp
            m = model.m + model.nrangeC + n_var_low + n_var_upp
            Lvar = np.zeros(n)
            Lvar[:model.n] = -np.inf * np.ones(model.n)
            Uvar = np.inf * np.ones(n)

        NLPModel.__init__(self, n=n, m=m, name='slack-' + model.name,
                          Lvar=Lvar, Uvar=Uvar, Lcon=np.zeros(m))

        # Redefine primal and dual initial guesses
        self.x0 = np.empty(self.n)
        self.x0[:model.n] = model.x0[:]
        self.x0[model.n:] = 0

        self.pi0 = np.empty(self.m)
        self.pi0[:model.m] = model.pi0[:]
        self.pi0[model.m:] = 0
        return

    def obj(self, x):
        """Evaluate the objective function at x."""
        return self.model.obj(x[:self.model.n])

    def grad(self, x):
        """Evaluate the objective gradient at x."""
        g = np.zeros(self.n)
        g[:self.model.n] = self.model.grad(x[:self.model.n])
        return g

    def cons(self, x):
        """Evaluate vector of constraints at x.

        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will be
        the constraint that has the slack on the lower bound on c[i]. The
        constraint with the slack on the upper bound on c[i] will be stored in
        position m + k, where k is the position of index i in rangeC, i.e., k=0
        iff constraint i is the range constraint that appears first, k=1 iff it
        appears second, etc.
        """
        model = self.model
        on = model.n
        m = self.m
        om = model.m
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC
        nrangeC = model.nrangeC

        c = np.empty(m)
        c[:om + nrangeC] = model.cons_pos(x[:on])
        c[lowerC] -= x[self.sLL]
        c[upperC] -= x[self.sUU]
        c[rangeC] -= x[self.sLR]
        c[om:] -= x[self.sUR]
        return c

    def cons_pos(self, x):
        """Convenience function to return constraints as non negative ones."""
        return self.cons(x)

    def jprod(self, x, p, **kwargs):
        """Evaluate Jacobian-vector product at x with p.

        Evaluate the Jacobian matrix-vector product of all equality
        constraints of the transformed problem with a vector `p` (J(x).T p).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = model.n
        om = model.m
        m = self.m

        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC
        nrangeC = model.nrangeC

        v = np.zeros(m)

        J = model.jop(x[:on])
        v[:om] = J * p[:on]
        v[upperC] *= -1.0
        v[om:om + nrangeC] = v[rangeC]
        v[om:om + nrangeC] *= -1.0

        # Insert contribution of slacks on general constraints
        v[lowerC] -= p[self.sLL]
        v[rangeC] -= p[self.sLR]
        v[upperC] -= p[self.sUU]
        v[om:om + nrangeC] -= p[self.sUR]

        if self.keep_variable_bounds:
            return v

        # Create some more shortcuts
        lowerB = model.lowerB
        upperB = model.upperB
        rangeB = model.rangeB
        nlowerB = model.nlowerB
        nupperB = model.nupperB
        nrangeB = model.nrangeB

        # Insert contribution of bound constraints on the original problem
        bot = om + nrangeC
        v[bot:bot + nlowerB] += p[lowerB]
        bot += nlowerB
        v[bot:bot + nrangeB] += p[rangeB]
        bot += nrangeB
        v[bot:bot + nupperB] -= p[upperB]
        bot += nupperB
        v[bot:bot + nrangeB] -= p[rangeB]

        # Insert contribution of slacks on the bound constraints
        bot = om + nrangeC
        v[bot:bot + nlowerB] -= p[self.tLL]
        bot += nlowerB
        v[bot:bot + nrangeB] -= p[self.tLR]
        bot += nrangeB
        v[bot:bot + nupperB] -= p[self.tUU]
        bot += nupperB
        v[bot:bot + nrangeB] -= p[self.tUR]
        return v

    def jtprod(self, x, p, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p.

        Evaluate the Jacobian-transpose matrix-vector product of all equality
        constraints of the transformed problem with a vector `p` (J(x).T p).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = model.n
        om = model.m
        n = self.n

        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC
        nrangeC = model.nrangeC

        v = np.zeros(n)
        pmv = p[:om].copy()
        pmv[upperC] *= -1.0
        pmv[rangeC] -= p[om:om + nrangeC]

        J = model.jop(x[:on])
        v[:on] = J.T * pmv

        # Insert contribution of slacks on general constraints
        v[self.sLL] = -p[lowerC]
        v[self.sLR] = -p[rangeC]
        v[self.sUU] = -p[upperC]
        v[self.sUR] = -p[om:om + nrangeC]

        if self.keep_variable_bounds:
            return v

        # Create some more shortcuts
        lowerB = model.lowerB
        upperB = model.upperB
        rangeB = model.rangeB
        nlowerB = model.nlowerB
        nupperB = model.nupperB
        nrangeB = model.nrangeB

        # Insert contribution of bound constraints on the original problem
        bot = om + nrangeC
        v[lowerB] += p[bot:bot + nlowerB]
        bot += nlowerB
        v[rangeB] += p[bot:bot + nrangeB]
        bot += nrangeB
        v[upperB] -= p[bot:bot + nupperB]
        bot += nupperB
        v[rangeB] -= p[bot:bot + nrangeB]

        # Insert contribution of slacks on the bound constraints
        bot = om + nrangeC
        v[self.tLL] -= p[bot:bot + nlowerB]
        bot += nlowerB
        v[self.tLR] -= p[bot:bot + nrangeB]
        bot += nrangeB
        v[self.tUU] -= p[bot:bot + nupperB]
        bot += nupperB
        v[self.tUR] -= p[bot:bot + nrangeB]
        return v

    def jac(self, x):
        """Evaluate constraints Jacobian at x.

        Evaluate the Jacobian matrix of all equality constraints of the
        transformed problem. The gradients of the general constraints appear in
        'natural' order, i.e., in the order in which they appear in the
        problem. The gradients of range constraints appear in two places: first
        in the 'natural' location and again after all other general
        constraints, with a flipped sign to account for the upper bound on
        those constraints.

        The gradients of the linear equalities corresponding to bounds on the
        original variables appear in the following order:

        1. variables with a lower bound only
        2. lower bound on variables with two-sided bounds
        3. variables with an upper bound only
        4. upper bound on variables with two-sided bounds

        The overall Jacobian of the new constraints thus has the form::

            [ J    -I         ]
            [-JR      -I      ]
            [ I          -I   ]
            [-I             -I]

        where the columns correspond, in order, to the variables `x`, `s`,
        `sU`, `t`, and `tU`, the rows correspond, in order, to

        1. general constraints (in natural order)
        2. 'upper' side of range constraints
        3. bounds, ordered as explained above
        4. 'upper' side of two-sided bounds,

        and where the signs corresponding to 'upper' constraints and upper
        bounds are flipped in the (1,1) and (3,1) blocks.
        """
        return self._jac(x, lp=False)

    def A(self):
        """Evaluate the constraints Jacobian of linear constraints at x.

        Return the constraint matrix if the problem is a linear program.
        See the documentation of :meth:`jac` for more information.
        """
        return self._jac(0, lp=True)

    def _jac(self, x, lp=False):
        raise NotImplementedError("Please subclass")

    def convert_multipliers(self, z):
        """Convert multipliers from slack problem to orginal NLP."""
        if z is None:
            return np.zeros(self.m)
        om = self.model.m
        upperC = self.model.upperC
        rangeC = self.model.rangeC
        nrangeC = self.model.nrangeC
        pi = z[:om].copy()
        pi[upperC] *= -1.
        pi[rangeC] -= z[om:om + nrangeC]
        return pi

    def hprod(self, x, y, v, **kwargs):
        """Hessian-vector product.

        Evaluate the Hessian vector product of the Hessian of the Lagrangian
        at (x,y) with the vector v.
        """
        model = self.model
        on = model.n

        Hv = np.zeros(self.n)
        pi = self.convert_multipliers(y)
        Hv[:on] = model.hprod(x[:on], pi, v[:on], **kwargs)
        return Hv

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        raise NotImplementedError("Please subclass")


class PySparseSlackModel(SlackModel):
    """Slack model in which matrices are represented as PySparse matrices."""

    def _jac(self, x, lp=False):
        """Helper method to assemble the Jacobian matrix of the constraints.

        See the documentation of :meth:`jac` for more information.

        The positional argument `lp` should be set to `True` only if the
        problem is known to be a linear program. In this case, the evaluation
        of the constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.n
        m = self.m
        model = self.model
        on = self.model.n
        om = self.model.m

        lowerC = np.array(model.lowerC)
        nlowerC = model.nlowerC
        upperC = np.array(model.upperC)
        nupperC = model.nupperC
        rangeC = np.array(model.rangeC)
        nrangeC = model.nrangeC
        lowerB = np.array(model.lowerB)
        nlowerB = model.nlowerB
        upperB = np.array(model.upperB)
        nupperB = model.nupperB
        rangeB = np.array(model.rangeB)
        nrangeB = model.nrangeB
        nbnds = nlowerB + nupperB + 2 * nrangeB
        nSlacks = nlowerC + nupperC + 2 * nrangeC

        # Initialize sparse Jacobian
        # Overestimate of the number of non zero elements
        nnzJ = 2 * self.model.nnzj + m + nrangeC + nbnds + nrangeB
        J = PysparseMatrix(nrow=m, ncol=n, sizeHint=nnzJ)

        # Insert contribution of general constraints.
        if lp:
            J[:om, :on] = self.model.A()
        else:
            J[:om, :on] = self.model.jac(x[:on])
        J[upperC, :on] *= -1.0
        J[om:om + nrangeC, :on] = J[rangeC, :on]  # upper side of range const.
        J[om:om + nrangeC, :on] *= -1.0

        # Create a few index lists
        rlowerC = np.array(range(nlowerC))
        rlowerB = np.array(range(nlowerB))
        rupperC = np.array(range(nupperC))
        rupperB = np.array(range(nupperB))
        rrangeC = np.array(range(nrangeC))
        rrangeB = np.array(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(-1.0, lowerC, on + rlowerC)
        J.put(-1.0, upperC, on + nlowerC + rupperC)
        J.put(-1.0, rangeC, on + nlowerC + nupperC + rrangeC)
        J.put(-1.0, om + rrangeC, on + nlowerC + nupperC + nrangeC + rrangeC)

        if self.keep_variable_bounds:
            return J

        # Insert contribution of bound constraints on the original problem
        bot = om + nrangeC
        J.put(1.0, bot + rlowerB, lowerB)
        bot += nlowerB
        J.put(1.0, bot + rrangeB, rangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, upperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, rangeB)

        # Insert contribution of slacks on the bound constraints
        bot = om + nrangeC
        J.put(-1.0, bot + rlowerB, on + nSlacks + rlowerB)
        bot += nlowerB
        J.put(-1.0, bot + rrangeB, on + nSlacks + nlowerB + rrangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, on + nSlacks + nlowerB + nrangeB + rupperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, on + nSlacks + nlowerB + nrangeB +
              nupperB + rrangeB)
        return J

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        model = self.model
        if isinstance(model, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        on = model.n
        pi = self.convert_multipliers(z)
        H = PysparseMatrix(nrow=self.n, ncol=self.n, symmetric=True,
                           sizeHint=self.model.nnzh)
        H[:on, :on] = self.model.hess(x[:on], pi, *args, **kwargs)
        return H


class RegQPInteriorPointSolver(object):
    u"""Solve a QP with the primal-dual-regularized interior-point method.

    Solve a convex quadratic program of the form::

       minimize    cᵀx + ½ xᵀ Q x
       subject to  A1 x + A2 s = b,                                 (QP)
                   s ≥ 0,

    where Q is a symmetric positive semi-definite matrix, the variables
    x are the original problem variables and s are slack variables. Any
    quadratic program may be converted to the above form by instantiation
    of the `SlackModel` class. The conversion to the slack formulation
    is mandatory in this implementation.

    The method is a variant of Mehrotra's predictor-corrector method where
    steps are computed by solving the primal-dual system in augmented form.

    Primal and dual regularization parameters may be specified by the user
    via the opional keyword arguments `regpr` and `regdu`. Both should be
    positive real numbers and should not be "too large". By default they
    are set to 1.0 and updated at each iteration.

    If `scale` is set to `True`, (QP) is scaled automatically prior to
    solution so as to equilibrate the rows and columns of the constraint
    matrix [A1 A2].

    Advantages of this method are that it is not sensitive to dense columns
    in A, no special treatment of the unbounded variables x is required,
    and a sparse symmetric quasi-definite system of equations is solved at
    each iteration. The latter, although indefinite, possesses a
    Cholesky-like factorization. Those properties makes the method
    typically more robust that a standard predictor-corrector
    implementation and the linear system solves are often much faster than
    in a traditional interior-point method in augmented form.
    """

    def __init__(self, qp, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale: Perform row and column equilibration of the constraint
                    matrix [A1 A2] prior to solution (default: `True`).

            :regpr: Initial value of primal regularization parameter
                    (default: `1.0`).

            :regdu: Initial value of dual regularization parameter
                    (default: `1.0`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :verbose: Turn on verbose mode (default `False`).
        """
        if not isinstance(qp, SlackModel):
            msg = 'Input problem must be an instance of SlackModel'
            raise ValueError(msg)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'cqp.solver')
        self.log = logging.getLogger(logger_name)

        self.verbose = kwargs.get('verbose', True)
        scale = kwargs.get('scale', True)

        self.qp = qp
        self.A = qp.A()               # Constraint matrix
        if not isinstance(self.A, PysparseMatrix):
            self.A = PysparseMatrix(matrix=self.A)

        _, n = self.A.shape
        on = qp.original_n
        # Record number of slack variables in QP
        self.nSlacks = qp.n - on

        # Collect basic info about the problem.
        zero = np.zeros(n)

        self.b = -qp.cons(zero)                  # Right-hand side
        self.c0 = qp.obj(zero)                   # Constant term in objective
        self.c = qp.grad(zero[:on])              # Cost vector
        self.Q = PysparseMatrix(matrix=qp.hess(zero[:on],
                                               np.zeros(qp.original_m)))

        # Apply in-place problem scaling if requested.
        self.prob_scaled = False
        if scale:
            self.t_scale = cputime()
            self.scale()
            self.t_scale = cputime() - self.t_scale
        else:
            # self.scale() sets self.normQ to the Frobenius norm of Q
            # and self.normA to the Frobenius norm of A as a by-product.
            # If we're not scaling, set normQ and normA manually.
            self.normQ = self.Q.matrix.norm('fro')
            self.normA = self.A.matrix.norm('fro')

        self.normb = norm_infty(self.b)
        self.normc = norm_infty(self.c)
        self.normbc = 1 + max(self.normb, self.normc)

        # Initialize augmented matrix.
        self.H = self.initialize_kkt_matrix()

        # It will be more efficient to keep the diagonal of Q around.
        self.diagQ = self.Q.take(range(qp.original_n))

        # We perform the analyze phase on the augmented system only once.
        # self.LBL will be initialized in solve().
        self.LBL = None

        # Set regularization parameters.
        self.regpr = kwargs.get('regpr', 1.0)
        self.regpr_min = 1.0e-8
        self.regdu = kwargs.get('regdu', 1.0)
        self.regdu_min = 1.0e-8

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 5)

        # Check input parameters.
        if self.regpr < 0.0:
            self.regpr = 0.0
        if self.regdu < 0.0:
            self.regdu = 0.0

        # Initialize format strings for display
        fmt_hdr = '%-4s  %9s' + '  %-8s' * 6 + \
                  '  %-7s  %-4s  %-4s' + '  %-8s' * 8
        self.header = fmt_hdr % ('Iter', 'Cost', 'pResid', 'dResid', 'cResid',
                                 'rGap', 'qNorm', 'rNorm', 'Mu', 'AlPr',
                                 'AlDu', 'LS Resid', 'RegPr', 'RegDu', 'Rho q',
                                 'Del r', 'Min(s)', 'Min(z)', 'Max(s)')
        self.format1 = '%-4d  %9.2e'
        self.format1 += '  %-8.2e' * 6
        self.format2 = '  %-7.1e  %-4.2f  %-4.2f'
        self.format2 += '  %-8.2e' * 8

        self.mu_history = []
        self.cond_history = []
        self.berr_history = []
        self.derr_history = []
        self.nrms_history = []
        self.lres_history = []

        self.condest = kwargs.get('condest', False)
        self.condest_history = []
        self.normest_history = []

        if self.verbose:
            self.display_stats()

        return

    def initialize_kkt_matrix(self):
        u"""Create and initialize KKT matrix.

        [ -(Q+ρI)      0             A1ᵀ ] [∆x]   [c + Q x - A1ᵀ y     ]
        [  0      -(S^{-1} Z + ρI)   A2ᵀ ] [∆s] = [- A2ᵀ y - µ S^{-1} e]
        [  A1          A2            δI  ] [∆y]   [b - A1 x - A2 s     ]
        """
        m, n = self.A.shape
        on = self.qp.original_n
        H = PysparseMatrix(size=n + m,
                           sizeHint=n + m + self.A.nnz + self.Q.nnz,
                           symmetric=True)

        # The (1,1) block will always be Q (save for its diagonal).
        H[:on, :on] = -self.Q

        # The (3,1) and (3,2) blocks will always be A.
        # We store it now once and for all.
        H[n:, :n] = self.A
        return H

    def initialize_rhs(self):
        """Initialize right-hand side."""
        m, n = self.A.shape
        return np.zeros(n + m)

    def set_affine_scaling_rhs(self, rhs, pFeas, dFeas, s, z):
        """Set rhs for affine-scaling step."""
        _, n = self.A.shape
        on = self.qp.original_n
        rhs[:n] = -dFeas
        rhs[on:n] += z
        rhs[n:] = -pFeas
        return

    def display_stats(self):
        """Display vital statistics about the input problem."""
        import os
        qp = self.qp
        log = self.log
        log.info('Problem Path: %s' % qp.name)
        log.info('Problem Name: %s' % os.path.basename(qp.name))
        log.info('Number of problem variables: %d' % qp.original_n)
        log.info('Number of free variables: %d' % qp.nfreeB)
        log.info('Number of problem constraints excluding bounds: %d' %
                 qp.original_m)
        log.info('Number of slack variables: %d' % (qp.n - qp.original_n))
        log.info('Adjusted number of variables: %d' % qp.n)
        log.info('Adjusted number of constraints excluding bounds: %d' % qp.m)
        log.info('Number of nonzeros in Hessian matrix Q: %d' % self.Q.nnz)
        log.info('Number of nonzeros in constraint matrix: %d' % self.A.nnz)
        log.info('Constant term in objective: %8.2e' % self.c0)
        log.info('Cost vector norm: %8.2e' % self.normc)
        log.info('Right-hand side norm: %8.2e' % self.normb)
        log.info('Hessian norm: %8.2e' % self.normQ)
        log.info('Jacobian norm: %8.2e' % self.normA)
        log.info('Initial primal regularization: %8.2e' % self.regpr)
        log.info('Initial dual   regularization: %8.2e' % self.regdu)
        if self.prob_scaled:
            log.info('Time for scaling: %6.2fs' % self.t_scale)
        return

    def scale(self, **kwargs):
        """Equilibrate the constraint matrix of the linear program.

        Equilibration is done by first dividing every row by its largest
        element in absolute value and then by dividing every column by its
        largest element in absolute value. In effect the original problem::

            minimize c' x + 1/2 x' Q x
            subject to  A1 x + A2 s = b, x >= 0

        is converted to::

            minimize (Cc)' x + 1/2 x' (CQC') x
            subject to  R A1 C x + R A2 C s = Rb, x >= 0,

        where the diagonal matrices R and C operate row and column scaling
        respectively.

        Upon return, the matrix A and the right-hand side b are scaled and the
        members `row_scale` and `col_scale` are set to the row and column
        scaling factors.

        The scaling may be undone by subsequently calling :meth:`unscale`. It
        is necessary to unscale the problem in order to unscale the final dual
        variables. Normally, the :meth:`solve` method takes care of unscaling
        the problem upon termination.
        """
        log = self.log
        m, n = self.A.shape
        row_scale = np.zeros(m)
        col_scale = np.zeros(n)
        (values, irow, jcol) = self.A.find()

        if self.verbose:
            log.info('Smallest and largest elements of A prior to scaling: ')
            log.info('%8.2e %8.2e' % (np.min(np.abs(values)),
                                      np.max(np.abs(values))))

        # Find row scaling.
        for k in range(len(values)):
            row = irow[k]
            val = abs(values[k])
            row_scale[row] = max(row_scale[row], val)
        row_scale[row_scale == 0.0] = 1.0

        if self.verbose:
            log.info('Max row scaling factor = %8.2e' % np.max(row_scale))

        # Apply row scaling to A and b.
        values /= row_scale[irow]
        self.b /= row_scale

        # Find column scaling.
        for k in range(len(values)):
            col = jcol[k]
            val = abs(values[k])
            col_scale[col] = max(col_scale[col], val)
        col_scale[col_scale == 0.0] = 1.0

        if self.verbose:
            log.info('Max column scaling factor = %8.2e' % np.max(col_scale))

        # Apply column scaling to A and c.
        values /= col_scale[jcol]
        self.c[:self.qp.original_n] /= col_scale[:self.qp.original_n]

        if self.verbose:
            log.info('Smallest and largest elements of A after scaling: ')
            log.info('%8.2e %8.2e' % (np.min(np.abs(values)),
                                      np.max(np.abs(values))))

        # Overwrite A with scaled values.
        self.A.put(values, irow, jcol)
        self.normA = norm2(values)   # Frobenius norm of A.

        # Apply scaling to Hessian matrix Q.
        (values, irow, jcol) = self.Q.find()
        values /= col_scale[irow]
        values /= col_scale[jcol]
        self.Q.put(values, irow, jcol)
        self.normQ = norm2(values)  # Frobenius norm of Q

        # Save row and column scaling.
        self.row_scale = row_scale
        self.col_scale = col_scale

        self.prob_scaled = True

        return

    def unscale(self, **kwargs):
        """Unscale the constraint matrix of the linear program.

        Restore the constraint matrix A, the right-hand side b and the cost
        vector c to their original value by undoing the row and column
        equilibration scaling.
        """
        row_scale = self.row_scale
        col_scale = self.col_scale
        on = self.qp.original_n

        # Unscale constraint matrix A.
        self.A.row_scale(row_scale)
        self.A.col_scale(col_scale)

        # Unscale right-hand side and cost vectors.
        self.b *= row_scale
        self.c[:on] *= col_scale[:on]

        # Unscale Hessian matrix Q.
        (values, irow, jcol) = self.Q.find()
        values *= col_scale[irow]
        values *= col_scale[jcol]
        self.Q.put(values, irow, jcol)

        # Recover unscaled multipliers y and z.
        self.y *= self.row_scale
        self.z /= self.col_scale[on:]

        self.prob_scaled = False

        return

    def solve(self, **kwargs):
        """Solve.

        Accepted input keyword arguments are

        :keywords:

          :itermax:  The maximum allowed number of iterations (default: 10n)
          :tolerance:  Stopping tolerance (default: 1.0e-6)
          :PredictorCorrector:  Use the predictor-corrector method
                                (default: `True`). If set to `False`, a variant
                                of the long-step method is used. The long-step
                                method is generally slower and less robust.

        :returns:

            :x:            final iterate
            :y:            final value of the Lagrange multipliers associated
                           to `A1 x + A2 s = b`
            :z:            final value of the Lagrange multipliers associated
                           to `s >= 0`
            :obj_value:    final cost
            :iter:         total number of iterations
            :kktResid:     final relative residual
            :solve_time:   time to solve the QP
            :status:       string describing the exit status.
            :short_status: short version of status, used for printing.

        """
        qp = self.qp
        itermax = kwargs.get('itermax', max(100, 10 * qp.n))
        tolerance = kwargs.get('tolerance', 1.0e-6)
        PredictorCorrector = kwargs.get('PredictorCorrector', True)
        check_infeasible = kwargs.get('check_infeasible', True)

        # Transfer pointers for convenience.
        on = qp.original_n
        A = self.A
        b = self.b
        c = self.c
        Q = self.Q
        H = self.H

        regpr = self.regpr
        regdu = self.regdu
        regpr_min = self.regpr_min
        regdu_min = self.regdu_min

        # Obtain initial point from Mehrotra's heuristic.
        (x, y, z) = self.set_initial_guess(**kwargs)

        # Slack variables are the trailing variables in x.
        s = x[on:]
        ns = self.nSlacks

        # Initialize steps in dual variables.
        dz = np.zeros(ns)

        # Allocate room for right-hand side of linear systems.
        rhs = self.initialize_rhs()
        finished = False
        iter = 0

        setup_time = cputime()

        # Main loop.
        while not finished:

            # Display initial header every so often.
            if iter % 50 == 0:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))

            # Compute residuals.
            pFeas = A * x - b
            comp = s * z
            sz = sum(comp)                # comp   = Sz
            Qx = Q * x[:on]
            dFeas = y * A
            dFeas[:on] -= self.c + Qx    # dFeas1 = A1'y - c - Qx
            dFeas[on:] += z                            # dFeas2 = A2'y + z

            # Compute duality measure.
            if ns > 0:
                mu = sz / ns
            else:
                mu = 0.0

            self.mu_history.append(mu)

            # Compute residual norms and scaled residual norms.
            pResid = norm2(pFeas)
            spResid = pResid / (1 + self.normb + self.normA + self.normQ)
            dResid = norm2(dFeas)
            sdResid = dResid / (1 + self.normc + self.normA + self.normQ)
            if ns > 0:
                cResid = norm_infty(comp) / (self.normbc + self.normA +
                                             self.normQ)
            else:
                cResid = 0.0

            # Compute relative duality gap.
            cx = np.dot(c, x[:on])
            xQx = np.dot(x[:on], Qx)
            by = np.dot(b, y)
            rgap = cx + xQx - by
            rgap = abs(rgap) / (1 + abs(cx) + self.normA + self.normQ)
            rgap2 = mu / (1 + abs(cx) + self.normA + self.normQ)

            # Compute overall residual for stopping condition.
            kktResid = max(spResid, sdResid, rgap2)

            # At the first iteration, initialize perturbation vectors
            # (q=primal, r=dual).
            # Should probably get rid of q when regpr=0 and of r when regdu=0.
            if iter == 0:
                if regpr > 0:
                    q = dFeas / regpr
                    qNorm = dResid / regpr
                    rho_q = dResid
                else:
                    q = dFeas
                    qNorm = dResid
                    rho_q = 0.0
                rho_q_min = rho_q
                if regdu > 0:
                    r = -pFeas / regdu
                    rNorm = pResid / regdu
                    del_r = pResid
                else:
                    r = -pFeas
                    rNorm = pResid
                    del_r = 0.0
                del_r_min = del_r
                pr_infeas_count = 0  # Used to detect primal infeasibility.
                du_infeas_count = 0  # Used to detect dual infeasibility.
                pr_last_iter = 0
                du_last_iter = 0
                mu0 = mu

            else:

                if regdu > 0:
                    regdu = regdu / 10
                    regdu = max(regdu, regdu_min)
                if regpr > 0:
                    regpr = regpr / 10
                    regpr = max(regpr, regpr_min)

                # Check for infeasible problem.
                if check_infeasible:
                    if mu < tolerance / 100 * mu0 and \
                            rho_q > 1. / tolerance / 1.0e+6 * rho_q_min:
                        pr_infeas_count += 1
                        if pr_infeas_count > 1 and pr_last_iter == iter - 1:
                            if pr_infeas_count > 6:
                                status = 'Problem seems to be (locally) dual'
                                status += ' infeasible'
                                short_status = 'dInf'
                                finished = True
                                continue
                        pr_last_iter = iter
                    else:
                        pr_infeas_count = 0

                    if mu < tolerance / 100 * mu0 and \
                            del_r > 1. / tolerance / 1.0e+6 * del_r_min:
                        du_infeas_count += 1
                        if du_infeas_count > 1 and du_last_iter == iter - 1:
                            if du_infeas_count > 6:
                                status = 'Problem seems to be (locally) primal'
                                status += ' infeasible'
                                short_status = 'pInf'
                                finished = True
                                continue
                        du_last_iter = iter
                    else:
                        du_infeas_count = 0

            # Display objective and residual data.
            output_line = self.format1 % (iter, cx + 0.5 * xQx, pResid,
                                          dResid, cResid, rgap, qNorm,
                                          rNorm)

            if kktResid <= tolerance:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if iter >= itermax:
                status = 'Maximum number of iterations reached'
                short_status = 'iter'
                finished = True
                continue

            # Record some quantities for display
            if ns > 0:
                mins = np.min(s)
                minz = np.min(z)
                maxs = np.max(s)
            else:
                mins = minz = maxs = 0

            # Compute augmented matrix and factorize it.

            factorized = False
            degenerate = False
            nb_bump = 0
            while not factorized and not degenerate:

                self.update_linear_system(s, z, regpr, regdu)
                self.log.debug('Factorizing')
                self.LBL.factorize(H)
                factorized = True

                # If the augmented matrix does not have full rank, bump up the
                # regularization parameters.
                if not self.LBL.isFullRank:
                    if self.verbose:
                        self.log.info('Primal-Dual Matrix Rank Deficient' +
                                      '... bumping up reg parameters')

                    if regpr == 0. and regdu == 0.:
                        degenerate = True
                    else:
                        if regpr > 0:
                            regpr *= 100
                        if regdu > 0:
                            regdu *= 100
                        nb_bump += 1
                        degenerate = nb_bump > self.bump_max
                    factorized = False

            # Abandon if regularization is unsuccessful.
            if not self.LBL.isFullRank and degenerate:
                status = 'Unable to regularize sufficiently.'
                short_status = 'degn'
                finished = True
                continue

            if PredictorCorrector:
                # Use Mehrotra predictor-corrector method.
                # Compute affine-scaling step, i.e. with centering = 0.
                self.set_affine_scaling_rhs(rhs, pFeas, dFeas, s, z)

                (step, nres, _) = self.solve_system(rhs)

                # Recover dx and dz.
                dx, ds, dy, dz = self.get_affine_scaling_dxsyz(step,
                                                               x, s, y, z)

                # Compute largest allowed primal and dual stepsizes.
                (alpha_p, ip) = self.max_step_length(s, ds)
                (alpha_d, ip) = self.max_step_length(z, dz)

                # Estimate duality gap after affine-scaling step.
                muAff = np.dot(s + alpha_p * ds, z + alpha_d * dz) / ns
                sigma = (muAff / mu)**3

                # Incorporate predictor information for corrector step.
                # Only update rhs[on:n]; the rest of the vector did not change.
                comp += ds * dz
                comp -= sigma * mu
                self.update_corrector_rhs(rhs, s, z, comp)
            else:
                # Use long-step method: Compute centering parameter.
                sigma = min(0.1, 100 * mu)
                comp -= sigma * mu

                # Assemble rhs.
                self.update_long_step_rhs(rhs, pFeas, dFeas, comp, s)

            # Solve augmented system.
            (step, nres, neig) = self.solve_system(rhs)

            # Recover step.
            dx, ds, dy, dz = self.get_dxsyz(step, x, s, y, z, comp)

            # Compute largest allowed primal and dual stepsizes.
            (alpha_p, ip) = self.max_step_length(s, ds)
            (alpha_d, id) = self.max_step_length(z, dz)

            # Compute fraction-to-the-boundary factor.
            tau = max(.9995, 1.0 - mu)

            if PredictorCorrector:
                # Compute actual stepsize using Mehrotra's heuristic.
                mult = 0.1

                # ip=-1 if ds ≥ 0, and id=-1 if dz ≥ 0
                if (ip != -1 or id != -1) and ip != id:
                    mu_tmp = np.dot(s + alpha_p * ds, z + alpha_d * dz) / ns

                if ip != -1 and ip != id:
                    zip = z[ip] + alpha_d * dz[ip]
                    gamma_p = ((mult * mu_tmp - s[ip] * zip) /
                               (alpha_p * ds[ip] * zip))
                    alpha_p *= max(1 - mult, gamma_p)

                if id != -1 and ip != id:
                    sid = s[id] + alpha_p * ds[id]
                    gamma_d = ((mult * mu_tmp - z[id] * sid) /
                               (alpha_d * dz[id] * sid))
                    alpha_d *= max(1 - mult, gamma_d)

                if ip == id and ip != -1:
                    # There is a division by zero in Mehrotra's heuristic
                    # Fall back on classical rule.
                    alpha_p *= tau
                    alpha_d *= tau

            else:
                alpha_p *= tau
                alpha_d *= tau

            # Display data.
            output_line += self.format2 % (mu, alpha_p, alpha_d,
                                           nres, regpr, regdu, rho_q,
                                           del_r, mins, minz, maxs)
            self.log.info(output_line)

            # Update iterates and perturbation vectors.
            x += alpha_p * dx    # This also updates slack variables.
            y += alpha_d * dy
            z += alpha_d * dz
            q *= (1 - alpha_p)
            q += alpha_p * dx
            r *= (1 - alpha_d)
            r += alpha_d * dy
            qNorm = norm2(q)
            rNorm = norm2(r)
            if regpr > 0:
                rho_q = regpr * qNorm / (1 + self.normc)
                rho_q_min = min(rho_q_min, rho_q)
            else:
                rho_q = 0.0
            if regdu > 0:
                del_r = regdu * rNorm / (1 + self.normb)
                del_r_min = min(del_r_min, del_r)
            else:
                del_r = 0.0
            iter += 1

        solve_time = cputime() - setup_time

        self.log.info('-' * len(self.header))

        # Transfer final values to class members.
        self.x = x
        self.y = y
        self.z = z
        self.iter = iter
        self.pResid = pResid
        self.cResid = cResid
        self.dResid = dResid
        self.rgap = rgap
        self.kktResid = kktResid
        self.solve_time = solve_time
        self.status = status
        self.short_status = short_status

        # Unscale problem if applicable.
        if self.prob_scaled:
            self.unscale()

        # Recompute final objective value.
        self.obj_value = self.c0 + cx + 0.5 * xQx

        return

    def set_initial_guess(self, **kwargs):
        """Compute initial guess according the Mehrotra's heuristic.

        Initial values of x are computed as the solution to the least-squares
        problem::

            minimize ||s||  subject to  A1 x + A2 s = b

        which is also the solution to the augmented system::

            [ Q   0   A1' ] [x]   [0]
            [ 0   I   A2' ] [s] = [0]
            [ A1  A2   0  ] [w]   [b].

        Initial values for (y,z) are chosen as the solution to the
        least-squares problem::

            minimize ||z||  subject to  A1' y = c,  A2' y + z = 0

        which can be computed as the solution to the augmented system::

            [ Q   0   A1' ] [w]   [c]
            [ 0   I   A2' ] [z] = [0]
            [ A1  A2   0  ] [y]   [0].

        To ensure stability and nonsingularity when A does not have full row
        rank, the (1,1) block is perturbed to 1.0e-4 * I and the (3,3) block is
        perturbed to -1.0e-4 * I.

        The values of s and z are subsequently adjusted to ensure they are
        positive. See [Methrotra, 1992] for details.
        """
        qp = self.qp
        n = qp.n
        on = qp.original_n

        self.log.debug('Computing initial guess')

        # Set up augmented system matrix and factorize it.
        self.set_initial_guess_system()
        self.LBL = LBLContext(self.H, sqd=self.regdu > 0)  # Analyze+factorize

        # Assemble first right-hand side and solve.
        rhs = self.set_initial_guess_rhs()
        (step, nres, neig) = self.solve_system(rhs)

        dx, _, _, _ = self.get_dxsyz(step, 0, 1, 0, 0, 0)

        # dx is just a reference; we need to make a copy.
        x = dx.copy()
        s = x[on:]  # Slack variables. Must be positive.

        # Assemble second right-hand side and solve.
        self.update_initial_guess_rhs(rhs)

        (step, nres, neig) = self.solve_system(rhs)

        _, dz, dy, _ = self.get_dxsyz(step, 0, 1, 0, 0, 0)

        # dy and dz are just references; we need to make copies.
        y = dy.copy()
        z = -dz

        # If there are no inequality constraints, this is it.
        if n == on:
            return (x, y, z)

        # Use Mehrotra's heuristic to ensure (s,z) > 0.
        if np.all(s >= 0):
            dp = 0.0
        else:
            dp = -1.5 * min(s[s < 0])
        if np.all(z >= 0):
            dd = 0.0
        else:
            dd = -1.5 * min(z[z < 0])

        if dp == 0.0:
            dp = 1.5
        if dd == 0.0:
            dd = 1.5

        es = sum(s + dp)
        ez = sum(z + dd)
        xs = sum((s + dp) * (z + dd))

        dp += 0.5 * xs / ez
        dd += 0.5 * xs / es
        s += dp
        z += dd

        if not np.all(s > 0) or not np.all(z > 0):
            raise ValueError('Initial point not strictly feasible')

        return (x, y, z)

    def max_step_length(self, x, d):
        """Compute step length to boundary from x in direction d.

        It computes the max step length from x to the boundary of the
        nonnegative orthant in the direction d. Also return the component index
        responsible for cutting the steplength the most (or -1 if no such index
        exists).
        """
        self.log.debug('Computing step length to boundary')
        whereneg = np.where(d < 0)[0]
        if len(whereneg) > 0:
            dxneg = -x[whereneg] / d[whereneg]
            kmin = np.argmin(dxneg)
            stepmax = min(1.0, dxneg[kmin])
            if stepmax == 1.0:
                kmin = -1
            else:
                kmin = whereneg[kmin]
        else:
            stepmax = 1.0
            kmin = -1
        return (stepmax, kmin)

    def set_initial_guess_system(self):
        """Set linear system for initial guess."""
        self.log.debug('Setting up linear system for initial guess')
        m, n = self.A.shape
        on = self.qp.original_n
        self.H.put(-self.diagQ - 1.0e-4, range(on))
        self.H.put(-1.0, range(on, n))
        self.H.put(1.0e-4, range(n, n + m))
        return

    def set_initial_guess_rhs(self):
        """Set right-hand side for initial guess."""
        self.log.debug('Setting up right-hand side for initial guess')
        m, n = self.A.shape
        rhs = np.zeros(n + m)
        rhs[n:] = self.b
        return rhs

    def update_initial_guess_rhs(self, rhs):
        """Update right-hand side for initial guess."""
        self.log.debug('Updating right-hand side for initial guess')
        on = self.qp.original_n
        rhs[:on] = self.c
        rhs[on:] = 0.0
        return

    def update_linear_system(self, s, z, regpr, regdu):
        """Update linear system for current iteration."""
        self.log.debug('Updating linear system for current iteration')
        qp = self.qp
        n = qp.n
        m = qp.m
        on = qp.original_n
        diagQ = self.diagQ
        self.H.put(-diagQ - regpr, range(on))
        self.H.put(-z / s - regpr, range(on, n))
        if regdu > 0:
            self.H.put(regdu, range(n, n + m))
        return

    def solve_system(self, rhs, itref_threshold=1.0e-5, nitrefmax=5):
        """Solve the augmented system with right-hand side `rhs`.

        It optionally performs iterative refinement.

        Return the solution vector (as a reference), the 2-norm of the residual
        and the number of negative eigenvalues of the coefficient matrix.
        """
        self.log.debug('Solving linear system')
        self.LBL.solve(rhs)
        self.LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)

        # Collect statistics on the linear system solve.
        self.cond_history.append((self.LBL.cond, self.LBL.cond2))
        self.berr_history.append((self.LBL.berr, self.LBL.berr2))
        self.derr_history.append(self.LBL.dirError)
        self.nrms_history.append((self.LBL.matNorm, self.LBL.xNorm))
        self.lres_history.append(self.LBL.relRes)

        # Estimate matrix l2-norm condition number.
        if self.condest:
            rhsNorm = norm2(rhs)
            solnNorm = norm2(self.LBL.x)
            Hop = PysparseLinearOperator(self.H, symmetric=True)
            normH, _ = normest(Hop, tol=1.0e-3)
            if rhsNorm > 0 and solnNorm > 0:
                self.condest_history.append(solnNorm * normH / rhsNorm)
            else:
                self.condest_history.append(1.)
            self.normest_history.append(normH)

        nr = norm2(self.LBL.residual)
        return (self.LBL.x, nr, self.LBL.neig)

    def get_affine_scaling_dxsyz(self, step, x, s, y, z):
        """Split `step` into steps along x, s, y and z.

        his function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        self.log.debug('Recovering affine-scaling step')
        m, n = self.A.shape
        on = self.qp.original_n
        dx = step[:n]
        ds = dx[on:]
        dy = step[n:]
        dz = -z * (1 + ds / s)
        return (dx, ds, dy, dz)

    def update_corrector_rhs(self, rhs, s, z, comp):
        """Update right-hand side for corrector step."""
        self.log.debug('Updating right-hand side for corrector step')
        m, n = self.A.shape
        on = self.qp.original_n
        rhs[on:n] += comp / s - z
        return

    def update_long_step_rhs(self, rhs, pFeas, dFeas, comp, s):
        """Update right-hand side when using long step."""
        self.log.debug('Updating right-hand side for long step')
        m, n = self.A.shape
        on = self.qp.original_n
        rhs[:n] = -dFeas
        rhs[on:n] += comp / s
        rhs[n:] = -pFeas
        return

    def get_dxsyz(self, step, x, s, y, z, comp):
        """Split `step` into steps along x, s, y and z.

        This function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        self.log.debug('Recovering step')
        m, n = self.A.shape
        on = self.qp.original_n
        dx = step[:n]
        ds = dx[on:]
        dy = step[n:]
        dz = -(comp + z * ds) / s
        return (dx, ds, dy, dz)


class RegQPInteriorPointSolver29(RegQPInteriorPointSolver):
    """A variant of the regularized interior-point method with MC29 scaling."""

    def scale(self, **kwargs):
        """Scale the constraint matrix of the linear program.

        The scaling is done so that the scaled matrix has all its entries near
        1.0 in the sense that the square of the sum of the logarithms of the
        entries is minimized.

        In effect the original problem::

            minimize c'x + 1/2 x'Qx subject to  A1 x + A2 s = b, x >= 0

        is converted to::

            minimize (Cc)'x + 1/2 x' (CQC') x
            subject to  R A1 C x + R A2 C s = Rb, x >= 0,

        where the diagonal matrices R and C operate row and column scaling
        respectively.

        Upon return, the matrix A and the right-hand side b are scaled and the
        members `row_scale` and `col_scale` are set to the row and column
        scaling factors.

        The scaling may be undone by subsequently calling :meth:`unscale`. It
        is necessary to unscale the problem in order to unscale the final dual
        variables. Normally, the :meth:`solve` method takes care of unscaling
        the problem upon termination.
        """
        (values, irow, jcol) = self.A.find()
        m, n = self.A.shape

        # Obtain row and column scaling
        row_scale, col_scale, ifail = mc29ad(m, n, values, irow, jcol)

        # row_scale and col_scale contain in fact the logarithms of the
        # scaling factors.
        row_scale = np.exp(row_scale)
        col_scale = np.exp(col_scale)

        # Apply row and column scaling to constraint matrix A.
        values *= row_scale[irow]
        values *= col_scale[jcol]

        # Overwrite A with scaled matrix.
        self.A.put(values, irow, jcol)

        # Apply row scaling to right-hand side b.
        self.b *= row_scale

        # Apply column scaling to cost vector c.
        self.c[:self.qp.original_n] *= col_scale[:self.qp.original_n]

        # Save row and column scaling.
        self.row_scale = row_scale
        self.col_scale = col_scale

        self.prob_scaled = True

        return

    def unscale(self, **kwargs):
        """Unscale the constraint matrix of the linear program.

        Restore the constraint matrix A, the right-hand side b and the cost
        vector c to their original value by undoing the row and column
        equilibration scaling.
        """
        row_scale = self.row_scale
        col_scale = self.col_scale
        on = self.qp.original_n

        # Unscale constraint matrix A.
        self.A.row_scale(1 / row_scale)
        self.A.col_scale(1 / col_scale)

        # Unscale right-hand side b.
        self.b /= row_scale

        # Unscale cost vector c.
        self.c[:on] /= col_scale[:on]

        # Recover unscaled multipliers y and z.
        self.y /= row_scale
        self.z *= col_scale[on:]

        self.prob_scaled = False

        return


class RegQPInteriorPointSolver3x3(RegQPInteriorPointSolver):
    """A 3x3 block variant of the regularized interior-point method.

    Linear system is based on the 3x3 block system instead of the reduced 2x2
    block system.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale: Perform row and column equilibration of the constraint
                    matrix [A1 A2] prior to solution (default: `True`).

            :regpr: Initial value of primal regularization parameter
                    (default: `1.0`).

            :regdu: Initial value of dual regularization parameter
                    (default: `1.0`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :verbose: Turn on verbose mode (default `False`).
        """
        super(RegQPInteriorPointSolver3x3, self).__init__(*args, **kwargs)

    def initialize_kkt_matrix(self):
        u"""Create and initialize KKT matrix.

        [ -(Q+ρI)   0        A1'     0   ]
        [  0       -ρI       A2'  Z^{1/2}]
        [  A1       A2       δI      0   ]
        [  0        Z^{1/2}  0       S   ]
        """
        m, n = self.A.shape
        on = self.qp.original_n
        H = PysparseMatrix(size=2 * n + m - on,
                           sizeHint=4 * on + m + self.A.nnz + self.Q.nnz,
                           symmetric=True)

        # The (1,1) block will always be Q (save for its diagonal).
        H[:on, :on] = -self.Q

        # The (2,1) block will always be A. We store it now once and for all.
        H[n:n + m, :n] = self.A
        return H

    def set_initial_guess_system(self):
        """Set linear system for initial guess."""
        m, n = self.A.shape
        on = self.qp.original_n
        self.H.put(-self.diagQ - 1.0e-4, range(on))
        self.H.put(-1.0, range(on, n))
        self.H.put(1.0e-4, range(n, n + m))
        self.H.put(1.0, range(n + m, 2 * n + m - on))
        self.H.put(1.0, range(n + m, 2 * n + m - on), range(on, n))
        return

    def set_initial_guess_rhs(self):
        """Set right-hand side for initial guess."""
        m, n = self.A.shape
        rhs = self.initialize_rhs()
        rhs[n:n + m] = self.b
        return rhs

    def update_initial_guess_rhs(self, rhs):
        """Update right-hand side for initial guess."""
        _, n = self.A.shape
        on = self.qp.original_n
        rhs[:on] = self.c
        rhs[on:] = 0.0
        return

    def initialize_rhs(self):
        """Initialize right-hand side with zeros."""
        m, n = self.A.shape
        on = self.qp.original_n
        return np.zeros(2 * n + m - on)

    def update_linear_system(self, s, z, regpr, regdu, **kwargs):
        """Update linear system."""
        qp = self.qp
        n = qp.n
        m = qp.m
        on = qp.original_n
        diagQ = self.diagQ
        self.H.put(-diagQ - regpr, range(on))
        if regpr > 0:
            self.H.put(-regpr, range(on, n))
        if regdu > 0:
            self.H.put(regdu, range(n, n + m))
        self.H.put(np.sqrt(z), range(n + m, 2 * n + m - on), range(on, n))
        self.H.put(s, range(n + m, 2 * n + m - on))
        return

    def set_affine_scaling_rhs(self, rhs, pFeas, dFeas, s, z):
        """Set rhs for affine-scaling step."""
        m, n = self.A.shape
        rhs[:n] = -dFeas
        rhs[n:n + m] = -pFeas
        rhs[n + m:] = -s * np.sqrt(z)
        return

    def get_affine_scaling_dxsyz(self, step, x, s, y, z):
        """Split `step` into steps along x, s, y and z.

        his function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        return self.get_dxsyz(step, x, s, y, z, 0)

    def update_corrector_rhs(self, rhs, s, z, comp):
        """Update right-hand side for corrector step."""
        m, n = self.A.shape
        rhs[n + m:] = -comp / np.sqrt(z)
        return

    def update_long_step_rhs(self, rhs, pFeas, dFeas, comp, s):
        """Update right-hand side when using long step."""
        m, n = self.A.shape
        rhs[:n] = -dFeas
        rhs[n:n + m] = -pFeas
        rhs[n + m:] = -comp
        return

    def get_dxsyz(self, step, x, s, y, z, comp):
        """Split `step` into steps along x, s, y and z.

        This function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        m, n = self.A.shape
        on = self.qp.original_n
        dx = step[:n]
        ds = step[on:n]
        dy = step[n:n + m]
        dz = np.sqrt(z) * step[n + m:]
        return (dx, ds, dy, dz)
