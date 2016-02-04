"""
A slack framework for NLPy.
"""
from nlpy.model.nlp import NLPModel
from nlpy.model.qnmodel import QuasiNewtonModel

import numpy as np


__docformat__ = 'restructuredtext'


class SlackModel(NLPModel):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

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

    :parameters:
        :nlp:  Original NLP to transform to a slack form.

    :keywords:
        :keep_variable_bounds: set to `True` if you don't want to convert
                               bounds on variables to inequalities. In this
                               case bounds are kept as they were in the
                               original NLP.

    """

    def __init__(self, nlp, keep_variable_bounds=False, **kwargs):

        self.nlp = nlp
        self.keep_variable_bounds = keep_variable_bounds

        n_con_low = nlp.nlowerC + nlp.nrangeC  # ineqs with lower bound.
        n_con_upp = nlp.nupperC + nlp.nrangeC  # ineqs with upper bound.
        n_var_low = nlp.nlowerB + nlp.nrangeB  # vars  with lower bound.
        n_var_upp = nlp.nupperB + nlp.nrangeB  # vars  with upper bound.
        n_slacks = n_con_low + n_con_upp       # slacks for constraints.

        bot = nlp.n;
        self.sLL = range(bot, bot + nlp.nlowerC); bot += nlp.nlowerC
        self.sLR = range(bot, bot + nlp.nrangeC); bot += nlp.nrangeC
        self.sUU = range(bot, bot + nlp.nupperC); bot += nlp.nupperC
        self.sUR = range(bot, bot + nlp.nrangeC)

        # Update effective number of variables and constraints
        if keep_variable_bounds:
            n = nlp.n + n_slacks
            m = nlp.m + nlp.nrangeC

            Lvar = np.zeros(n)
            Lvar[:nlp.n] = nlp.Lvar
            Uvar = np.inf * np.ones(n)
            Uvar[:nlp.n] = nlp.Uvar

        else:
            bot += nlp.nrangeC;
            self.tLL = range(bot, bot + nlp.nlowerB); bot += nlp.nlowerB
            self.tLR = range(bot, bot + nlp.nrangeB); bot += nlp.nrangeB
            self.tUU = range(bot, bot + nlp.nupperB); bot += nlp.nupperB
            self.tUR = range(bot, bot + nlp.nrangeB)

            n = nlp.n + n_con_low + n_con_upp + n_var_low + n_var_upp
            m = nlp.m + nlp.nrangeC + n_var_low + n_var_upp
            Lvar = np.zeros(n)
            Lvar[:nlp.n] = -np.inf * np.ones(nlp.n)
            Uvar = np.inf * np.ones(n)

        NLPModel.__init__(self, n=n, m=m, name='slack-' + nlp.name,
                          Lvar=Lvar, Uvar=Uvar, Lcon=np.zeros(m))

        # Redefine primal and dual initial guesses
        self.x0 = np.empty(self.n)
        self.x0[:nlp.n] = nlp.x0[:]
        self.x0[nlp.n:] = 0

        self.pi0 = np.empty(self.m)
        self.pi0[:nlp.m] = nlp.pi0[:]
        self.pi0[nlp.m:] = 0
        return

    def obj(self, x):
        """
        Return the value of the objective function at `x`.
        """
        return self.nlp.obj(x[:self.nlp.n])

    def grad(self, x):
        """
        Return the value of the gradient of the objective function at `x`.
        """
        g = np.zeros(self.n)
        g[:self.nlp.n] = self.nlp.grad(x[:self.nlp.n])
        return g

    def cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.
        """
        nlp = self.nlp; on = nlp.n; m = self.m; om = nlp.m
        lowerC = nlp.lowerC; upperC = nlp.upperC
        rangeC = nlp.rangeC; nrangeC = nlp.nrangeC

        c = np.empty(m)
        c[:om + nrangeC] = nlp.cons_pos(x[:on])
        c[lowerC] -= x[self.sLL]
        c[upperC] -= x[self.sUU]
        c[rangeC] -= x[self.sLR]
        c[om:]    -= x[self.sUR]
        return c

    def cons_pos(self, x):
        return self.cons(x)

    def jprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x).T v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = nlp.n; om = nlp.m; m = self.m
        lowerC = nlp.lowerC; upperC = nlp.upperC; rangeC = nlp.rangeC
        nrangeC = nlp.nrangeC

        p = np.zeros(m)

        J = nlp.jop(x[:on])
        p[:om] = J * v[:on]
        p[upperC] *= -1.0
        p[om:om+nrangeC] = p[rangeC]
        p[om:om+nrangeC] *= -1.0

        # Insert contribution of slacks on general constraints
        p[lowerC] -= v[self.sLL]
        p[rangeC] -= v[self.sLR]
        p[upperC] -= v[self.sUU]
        p[om:om+nrangeC] -= v[self.sUR]

        if self.keep_variable_bounds:
            return p

        # Create some more shortcuts
        lowerB = nlp.lowerB ; upperB = nlp.upperB ; rangeB = nlp.rangeB
        nlowerB = nlp.nlowerB ; nupperB = nlp.nupperB ; nrangeB = nlp.nrangeB

        # Insert contribution of bound constraints on the original problem
        bot = om+nrangeC; p[bot:bot+nlowerB] += v[lowerB]
        bot += nlowerB;   p[bot:bot+nrangeB] += v[rangeB]
        bot += nrangeB;   p[bot:bot+nupperB] -= v[upperB]
        bot += nupperB;   p[bot:bot+nrangeB] -= v[rangeB]

        # Insert contribution of slacks on the bound constraints
        bot = om+nrangeC; p[bot:bot+nlowerB] -= v[self.tLL]
        bot += nlowerB;   p[bot:bot+nrangeB] -= v[self.tLR]
        bot += nrangeB;   p[bot:bot+nupperB] -= v[self.tUU]
        bot += nupperB;   p[bot:bot+nrangeB] -= v[self.tUR]
        return p

    def jtprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian-transpose matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x).T v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = nlp.n ; om = nlp.m ; n = self.n
        lowerC = nlp.lowerC ; upperC = nlp.upperC ; rangeC = nlp.rangeC
        nrangeC = nlp.nrangeC;

        p = np.zeros(n)
        vmp = v[:om].copy()
        vmp[upperC] *= -1.0
        vmp[rangeC] -= v[om:om+nrangeC]

        J = nlp.jop(x[:on])
        p[:on] = J.T * vmp

        # Insert contribution of slacks on general constraints
        p[self.sLL] = -v[lowerC]
        p[self.sLR] = -v[rangeC]
        p[self.sUU] = -v[upperC]
        p[self.sUR] = -v[om:om+nrangeC]

        if self.keep_variable_bounds:
            return p

        # Create some more shortcuts
        lowerB = nlp.lowerB ; upperB = nlp.upperB ; rangeB = nlp.rangeB
        nlowerB = nlp.nlowerB ; nupperB = nlp.nupperB ; nrangeB = nlp.nrangeB

        # Insert contribution of bound constraints on the original problem
        bot = om+nrangeC; p[lowerB] += v[bot:bot+nlowerB]
        bot += nlowerB;   p[rangeB] += v[bot:bot+nrangeB]
        bot += nrangeB;   p[upperB] -= v[bot:bot+nupperB]
        bot += nupperB;   p[rangeB] -= v[bot:bot+nrangeB]

        # Insert contribution of slacks on the bound constraints
        bot = om+nrangeC; p[self.tLL] -= v[bot:bot+nlowerB]
        bot += nlowerB;   p[self.tLR] -= v[bot:bot+nrangeB]
        bot += nrangeB;   p[self.tUU] -= v[bot:bot+nupperB]
        bot += nupperB;   p[self.tUR] -= v[bot:bot+nrangeB]
        return p

    def jac(self, x):
        """
        Evaluate the Jacobian matrix of all equality constraints of the
        transformed problem. The gradients of the general constraints appear in
        'natural' order, i.e., in the order in which they appear in the problem.
        The gradients of range constraints appear in two places: first in the
        'natural' location and again after all other general constraints, with a
        flipped sign to account for the upper bound on those constraints.

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

        where the columns correspond, in order, to the variables `x`, `s`, `sU`,
        `t`, and `tU`, the rows correspond, in order, to

        1. general constraints (in natural order)
        2. 'upper' side of range constraints
        3. bounds, ordered as explained above
        4. 'upper' side of two-sided bounds,

        and where the signs corresponding to 'upper' constraints and upper
        bounds are flipped in the (1,1) and (3,1) blocks.
        """
        return self._jac(x, lp=False)

    def A(self):
        """
        Return the constraint matrix if the problem is a linear program. See the
        documentation of :meth:`jac` for more information.
        """
        return self._jac(0, lp=True)

    def jac_pos(self, x, **kwargs):
        return self.jac(x, **kwargs)

    def _jac(self, x, lp=False):
        raise NotImplementedError("Please subclass")

    def convert_multipliers(self, z):
        """
        Transform multipliers for the slack problem into multipliers for the
        original NLP.
        """
        if z is None:
            return np.zeros(self.m)
        om = self.nlp.m; upperC = self.nlp.upperC
        rangeC = self.nlp.rangeC; nrangeC = self.nlp.nrangeC
        pi = z[:om].copy()
        pi[upperC] *= -1.
        pi[rangeC] -= z[om:om + nrangeC]
        return pi

    def hprod(self, x, y, v, **kwargs):
        """
        Evaluate the Hessian vector product of the Hessian of the Lagrangian
        at (x,y) with the vector v.
        """
        nlp = self.nlp; on = nlp.n

        Hv = np.zeros(self.n)
        pi = self.convert_multipliers(y)
        Hv[:on] = nlp.hprod(x[:on], pi, v[:on], **kwargs)
        return Hv

    def hess(self, x, z=None, *args, **kwargs):
        raise NotImplementedError("Please subclass")



