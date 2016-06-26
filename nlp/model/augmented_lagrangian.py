# -*- coding: utf-8 -*-
"""An augmented Lagrangian model."""

from nlp.model.nlpmodel import NLPModel, BoundConstrainedNLPModel
from nlp.model.qnmodel import QuasiNewtonModel
from nlp.model.snlp import SlackModel

import numpy as np


class AugmentedLagrangian(BoundConstrainedNLPModel):
    u"""A bound-constrained proximal augmented Lagrangian.

    In case the original NLP has general inequalities, slack variables are
    introduced.

    The proximal augmented Lagrangian is defined as:

        L(x, π; ρ) := f(x) - πᵀc(x) + ½ δ ‖c(x)‖² + ½ ρ ‖x - xₖ‖²,

    where π are the current Lagrange multiplier estimates, δ is the
    current penalty parameter, ρ is the current proximal parameter and xₖ is
    a fixed vector.
    """

    def __init__(self, model, **kwargs):
        """Initialize a proximal augmented Lagrangian model from an `NLPModel`.

        :parameters:
            :model:   original NLPModel.

        :keywords:
            :penalty:  initial value for the penalty parameter (default: 10)
            :prox:     initial value for the proximal parameter (default: 0)
            :pi:       vector of initial multipliers (default: model.pi0)
            :xk:       initial value of the proximal vector (default: all zero)
        """
        if not isinstance(model, NLPModel):
            raise TypeError("model should be a subclass of NLPModel")

        if not isinstance(model, SlackModel):
            self.model = SlackModel(model, **kwargs)

        super(AugmentedLagrangian, self).__init__(self.model.n,
                                                  name='Al-' + self.model.name,
                                                  Lvar=self.model.Lvar,
                                                  Uvar=self.model.Uvar)


        self.penalty_init = kwargs.get('penalty', 10.)
        self._penalty = self.penalty_init

        self.prox_init = kwargs.get("prox", 0.)
        self._prox = self.prox_init

        self.pi0 = np.zeros(self.model.m)
        self.pi = self.pi0.copy()
        self.x0 = self.model.x0

        self.xk = kwargs.get("xk",
                             np.zeros(self.n) if self.prox_init > 0 else None)

    @property
    def penalty(self):
        """Current penalty parameter."""
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self._penalty = max(0, value)
        self.logger.debug("setting penalty parameter to %7.1e", self.penalty)

    @property
    def prox(self):
        """Current proximal parameter."""
        return self._prox

    @prox.setter
    def prox(self, value):
        self._prox = max(0, value)
        self.logger.debug("setting prox parameter to %7.1e", self.prox)

    def obj(self, x, **kwargs):
        """Evaluate augmented Lagrangian."""
        cons = self.model.cons(x)

        alfunc = self.model.obj(x)
        alfunc -= np.dot(self.pi, cons)
        alfunc += 0.5 * self.penalty * np.dot(cons, cons)
        if self.prox > 0:
            alfunc += 0.5 * self.prox * np.linalg.norm(x - self.xk)**2
        return alfunc

    def grad(self, x, **kwargs):
        """Evaluate augmented Lagrangian gradient."""
        model = self.model
        J = model.jop(x)
        cons = model.cons(x)
        algrad = model.grad(x) + J.T * (self.penalty * cons - self.pi)
        if self.prox > 0:
            algrad += self.prox * (x - self.xk)
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

        w = model.hprod(x, self.pi - self.penalty * cons, v)
        J = model.jop(x)
        Hv = w + self.penalty * J.T * J * v
        if self.prox > 0:
            Hv += self.prox * v
        return Hv

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
