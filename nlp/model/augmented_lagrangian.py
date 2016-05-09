# -*- coding: utf-8 -*-
"""An augmented Lagrangian model."""

from nlp.model.nlpmodel import NLPModel, BoundConstrainedNLPModel
from nlp.model.qnmodel import QuasiNewtonModel
from nlp.model.snlp import SlackModel

import numpy as np


class AugmentedLagrangian(BoundConstrainedNLPModel):
    u"""A bound-constrained augmented Lagrangian.

    In case the original NLP has general inequalities, slack variables are
    introduced.

    The augmented Lagrangian is defined as:

        L(x, π; ρ) := f(x) - πᵀc(x) + ½ ρ ‖c(x)‖².

    where π are the current Lagrange multiplier estimates and ρ is the
    current penalty parameter.
    """

    def __init__(self, model, **kwargs):
        """Initialize an augmented Lagrangian model from an `NLPModel`.

        :parameters:

            :model:   original NLPModel.

        :keywords:

            :rho:  initial value for the penalty parameter (default: 10.)
            :pi:   vector of initial multipliers (default: all zero.)
        """
        if not isinstance(model, NLPModel):
            raise TypeError("model should be a subclass of NLPModel")

        if not isinstance(model, SlackModel):
            self.model = SlackModel(model, **kwargs)

        super(AugmentedLagrangian, self).__init__(self.model.n,
                                                  name='Al-' + self.model.name,
                                                  Lvar=self.model.Lvar,
                                                  Uvar=self.model.Uvar)

        self.rho_init = kwargs.get('rho', 10.)
        self._rho = self.rho_init

        self.pi0 = np.zeros(self.model.m)
        self.pi = self.pi0.copy()
        self.x0 = self.model.x0

    @property
    def rho(self):
        """Current penalty parameter."""
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = max(0, value)

    def obj(self, x, **kwargs):
        """Evaluate augmented Lagrangian."""
        cons = self.model.cons(x)

        alfunc = self.model.obj(x)
        alfunc -= np.dot(self.pi, cons)
        alfunc += 0.5 * self.rho * np.dot(cons, cons)
        return alfunc

    def grad(self, x, **kwargs):
        """Evaluate augmented Lagrangian gradient."""
        model = self.model
        J = model.jop(x)
        cons = model.cons(x)
        algrad = model.grad(x) + J.T * (self.rho * cons - self.pi)
        return algrad

    def dual_feasibility(self, x, **kwargs):
        """Evaluate Lagrangian gradient."""
        model = self.model
        J = model.jop(x)
        lgrad = model.grad(x) - J.T * self.pi
        return lgrad

    def hprod(self, x, z, v, **kwargs):
        """Hessian-vector product.

        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with a vector v.
        """
        model = self.model
        cons = model.cons(x)

        w = model.hprod(x, self.pi - self.rho * cons, v)
        J = model.jop(x)
        return w + self.rho * J.T * J * v

    def hess(self, *args, **kwargs):
        """Obtain Lagrangian Hessian as a linear operator.

        See :meth:`NLPModel.hprod` for more information.
        """
        return self.hop(*args, **kwargs)


class QuasiNewtonAugmentedLagrangian(QuasiNewtonModel, AugmentedLagrangian):
    """Bound-constrained augmented Lagrangian with quasi-Newton Hessian.

    In instances of this class, the quasi-Newton Hessian approximates the
    Hessian of the augmented Lagrangian as a whole.

    If the quasi-Newton Hessian should approximate only the Hessian of the
    Lagrangian, consider an initialization of the form

            AugmentedLagrangian(QuasiNewtonModel(...))
    """

    pass  # All the work is done by the parent classes.
