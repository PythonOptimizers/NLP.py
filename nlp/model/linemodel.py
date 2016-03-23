# -*- coding: utf-8 -*-
"""Restriction of models to lines."""

from nlp.model.nlpmodel import NLPModel
import numpy as np


class C1LineModel(NLPModel):
    u"""Restriction of a C¹ model to a line.

    An instance of this class is a model representing the original model
    restricted to the line x + td. More precisely, if the original objective
    is f: ℝⁿ → ℝ, then, given x ∈ ℝⁿ and d ∈ ℝⁿ (d≠0) a fixed direction, the
    objective of the restricted model is ϕ: ℝ → ℝ defined by

        ϕ(t) := f(x + td).

    Similarly, if the original constraints are c: ℝⁿ → ℝᵐ, the constraints of
    the restricted model are γ: ℝ → ℝᵐ defined by

        γ(t) := c(x + td).

    The functions f and c are only assumed to be C¹, i.e., only values and
    first derivatives of ϕ and γ are defined.
    """

    def __init__(self, model, x, d):
        """Instantiate the restriction of a model to the line x + td.

        :parameters:
            :model: ```NLPModel``` whose objective is to be restricted
            :x: Numpy array
            :d: Numpy array assumed to be nonzero (no check is performed).
        """
        name = "line-" + model.name
        super(C1LineModel, self).__init__(1,
                                          m=model.ncon,
                                          name=name,
                                          x0=0,
                                          Lvar=model.Lvar,
                                          Uvar=model.Uvar,
                                          Lcon=model.Lcon,
                                          Ucon=model.Ucon)
        self.__x = x
        self.__d = d
        self.__model = model

    @property
    def x(self):
        return self.__x

    @property
    def d(self):
        return self.__d

    @property
    def dir(self):
        return self.__d

    @property
    def model(self):
        return self.__model

    def obj(self, t):
        u"""Evaluate ϕ(t) = f(x + td)."""
        return self.model.obj(self.x + t * self.d)

    def grad(self, t):
        u"""Evaluate ϕ'(t) = ∇f(x + td)ᵀ d."""
        return np.dot(self.model.grad(self.x + t * self.d), self.d)

    def cons(self, t):
        u"""Evaluate γ(t) = c(x + td)."""
        return self.model.cons(self.x + t * self.d)

    def jac(self, t):
        u"""Evaluate γ'(t) = J(x + td) d."""
        return self.model.jprod(self.x + t * self.d, self.d)

    def jprod(self, t, v):
        u"""Jacobian-vector product, which is just γ'(t)*v with v ∈ ℝ."""
        return self.jac(t) * v

    def jtprod(self, t, u):
        u"""Transposed-Jacobian-vector product γ'(t)ᵀ u with u ∈ ℝᵐ."""
        return np.dot(self.jac(t), u)


class C2LineModel(C1LineModel):
    u"""Restriction of a C² objective function to a line.

    If f: ℝⁿ → ℝ, x ∈ ℝⁿ and d ∈ ℝⁿ (d≠0) is a fixed direction, an instance
    of this class is a model representing the function f restricted to
    the line x + td, i.e., the function ϕ: ℝ → ℝ defined by

        ϕ(t) := f(x + td).

    The function f is assumed to be C², i.e., values and first and second
    derivatives of ϕ are defined.
    """

    def hess(self, t, z):
        u"""Evaluate ϕ"(t) = dᵀ ∇²L(x + td, z) d."""
        return np.dot(self.d, self.model.hprod(self.x + t * self.d, z, self.d))

    def hprod(self, t, z, v):
        u"""Hessian-vector product, which is just ϕ"(t)*v with v ∈ ℝ."""
        return self.hess(t, z) * v
