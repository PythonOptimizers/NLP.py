# -*- coding: utf-8 -*-
"""A slack framework for NLP.py."""


import numpy as np
from nlp.model.nlpmodel import NLPModel

__docformat__ = 'restructuredtext'


class SlackModel(NLPModel):
    u"""General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    Original problem::

         cᴸ ≤ c(x)
              c(x) ≤ cᵁ
        cᴿᴸ ≤ c(x) ≤ cᴿᵁ
              c(x) = cᴱ
          l ≤   x  ≤ u

    is transformed to::

        c(x) - sᴸ = 0
        c(x) - sᵁ = 0
        c(x) - sᴿ = 0
        c(x) - cᴱ = 0

         cᴸ ≤ sᴸ
              sᵁ ≤ cᵁ
        cᴿᴸ ≤ sᴿ ≤ cᴿᵁ
          l ≤ x  ≤ u

    In the latter problem, the only inequality constraints are bounds on the
    slack and original variables. The other constraints are (typically)
    nonlinear equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sᴸ, the slack variables corresponding to general constraints with
       a lower bound only.

    3. sᵁ, the slack variables corresponding to general constraints with
       an upper bound only.

    4. sᴿ, the slack variables corresponding to general constraints with
       a lower bound and an upper bound.

    This framework initializes the slack variables sL and sU to
    zero by default.

    Note that the slack framework does not update all members of NLPModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, model, **kwargs):
        """Initialize a slack form of an :class:`NLPModel`.

        :parameters:
            :model:  Original model to be transformed into a slack form.
        """
        self.model = model

        # Save number of variables and constraints prior to transformation
        self.original_n = model.n
        self.original_m = model.m

        # Number of slacks for the constaints
        n_slacks = model.nlowerC + model.nupperC + model.nrangeC
        self.n_slacks = n_slacks

        # Update effective number of variables and constraints
        n = self.original_n + n_slacks
        m = self.original_m + model.nrangeC

        Lvar = -np.infty * np.ones(n)
        Uvar = +np.infty * np.ones(n)

        # Copy orignal bounds
        Lvar[:self.original_n] = model.Lvar
        Uvar[:self.original_n] = model.Uvar

        # Add bounds corresponding to lower constraints
        bot = self.original_n
        self.sL = range(bot, bot + model.nlowerC)
        Lvar[bot:bot + model.nlowerC] = model.Lcon[model.lowerC]

        # Add bounds corresponding to upper constraints
        bot += model.nlowerC
        self.sU = range(bot, bot + model.nupperC)
        Uvar[bot:bot + model.nupperC] = model.Ucon[model.upperC]

        # Add bounds corresponding to range constraints
        bot += model.nupperC
        self.sR = range(bot, bot + model.nrangeC)
        Lvar[bot:bot + model.nrangeC] = model.Lcon[model.rangeC]
        Uvar[bot:bot + model.nrangeC] = model.Ucon[model.rangeC]

        # No more inequalities. All constraints are now equal to 0
        Lcon = Ucon = np.zeros(m)

        super(SlackModel, self).__init__(n=n, m=m, name='Slack-' + model.name,
                                         Lvar=Lvar, Uvar=Uvar,
                                         Lcon=Lcon, Ucon=Ucon)

        # Redefine primal and dual initial guesses
        self.original_x0 = model.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = model.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]
        return

    def initialize_slacks(self, val=0.0, **kwargs):
        """Initialize all slack variables to given value.

        This method may need to be overridden.
        """
        self.x0[self.original_n:] = val
        return

    def obj(self, x):
        """Evaluate the objective function at x..

        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        f = self.model.obj(x[:self.original_n])

        return f

    def grad(self, x):
        """Evaluate the objective gradient at x.

        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        g = np.zeros(self.n)
        g[:self.original_n] = self.model.grad(x[:self.original_n])

        return g

    def cons(self, x):
        """Evaluate vector of constraints at x.

        Constraints are stored in the order in which they appear in the
        original problem.
        """
        on = self.original_n
        model = self.model

        equalC = model.equalC
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        c = model.cons(x[:on])

        c[equalC] -= model.Lcon[equalC]
        c[lowerC] -= x[self.sL]
        c[upperC] -= x[self.sU]
        c[rangeC] -= x[self.sR]

        return c

    def jprod(self, x, v, **kwargs):
        """Evaluate Jacobian-vector product at x with p.

        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        p = model.jprod(x[:on], v[:on])

        # Insert contribution of slacks on general constraints
        p[lowerC] -= v[self.sL]
        p[upperC] -= v[self.sU]
        p[rangeC] -= v[self.sR]
        return p

    def jtprod(self, x, v, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p.

        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n
        n = self.n
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        p = np.zeros(n)
        p[:on] = model.jtprod(x[:on], v)

        # Insert contribution of slacks on general constraints
        p[self.sL] = -v[lowerC]
        p[self.sU] = -v[upperC]
        p[self.sR] = -v[rangeC]
        return p

    def _jac(self, x, lp=False):
        """Helper method to assemble the Jacobian matrix of the constraints.

        See the documentation of :meth:`jac` for more information.

        The positional argument `lp` should be set to `True` only if the
        problem is known to be a linear program. In this case, the evaluation
        of the constraint matrix is cheaper and the argument `x` is ignored.
        """
        raise NotImplementedError("Please subclass")

    def jac(self, x):
        """Evaluate constraints Jacobian at x.

        The gradients of the general constraints appear in 'natural' order,
        i.e., in the order in which they appear in the problem.

        The overall Jacobian of the  constraints has the form::

            [ J    -I ]

        where the columns correspond to the variables `x` and `s`, and
        the rows correspond to the general constraints (in natural order).

        """
        return self._jac(x, lp=False)

    def A(self):
        """Return the constraint matrix if the problem is a linear program.

        See the documentation of :meth:`jac` for more information.
        """
        return self._jac(0, lp=True)

    def hprod(self, x, y, v, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        if y is None:
            y = np.zeros(self.m)

        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n

        Hv = np.zeros(self.n)
        Hv[:on] = model.hprod(x[:on], y, v[:on], **kwargs)
        return Hv

    def hess(self, x, z=None, *args, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        raise NotImplementedError("Please subclass")

    def ghivprod(self, x, g, v, **kwargs):
        """Evaluate individual dot products (g, Hi(x)*v).

        Evaluate the vector of dot products (g, Hi(x)*v) where Hi(x) is the
        Hessian of the i-th constraint at point x, i=1..m.
        """
        # Some shortcuts for convenience
        model = self.model
        on = self.original_n
        om = self.original_m

        gHiv = np.zeros(self.m)
        gHiv[:om] = model.ghivprod(x[:on], g[:on], v[:on], **kwargs)
        return gHiv
