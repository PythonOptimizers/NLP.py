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
        self.__f = None  # most recent objective value of `model`
        self.__g = None  # most recent objective gradient of `model`
        self.__c = None  # most recent constraint values of `model`
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
    def objval(self):
        return self.__f

    @property
    def gradval(self):
        return self.__g

    @property
    def conval(self):
        return self.__c

    @property
    def model(self):
        return self.__model

    def obj(self, t, x=None):
        u"""Evaluate ϕ(t) = f(x + td).

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        xtd = (self.x + t * self.d) if x is None else x
        self.__f = self.model.obj(xtd)
        return self.objval

    def grad(self, t, x=None):
        u"""Evaluate ϕ'(t) = ∇f(x + td)ᵀ d.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        xtd = (self.x + t * self.d) if x is None else x
        self.__g = self.model.grad(xtd)
        return np.dot(self.gradval, self.d)

    def cons(self, t, x=None):
        u"""Evaluate γ(t) = c(x + td).

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        xtd = (self.x + t * self.d) if x is None else x
        self.__c = self.model.cons(xtd)
        return self.conval

    def jac(self, t, x=None):
        u"""Evaluate γ'(t) = J(x + td) d.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        xtd = (self.x + t * self.d) if x is None else x
        return self.model.jprod(xtd, self.d)

    def jprod(self, t, v, x=None):
        u"""Jacobian-vector product, which is just γ'(t)*v with v ∈ ℝ.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        return self.jac(t, x=x) * v

    def jtprod(self, t, u, x=None):
        u"""Transposed-Jacobian-vector product γ'(t)ᵀ u with u ∈ ℝᵐ.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        return np.dot(self.jac(t, x=x), u)


class C2LineModel(C1LineModel):
    u"""Restriction of a C² objective function to a line.

    If f: ℝⁿ → ℝ, x ∈ ℝⁿ and d ∈ ℝⁿ (d≠0) is a fixed direction, an instance
    of this class is a model representing the function f restricted to
    the line x + td, i.e., the function ϕ: ℝ → ℝ defined by

        ϕ(t) := f(x + td).

    The function f is assumed to be C², i.e., values and first and second
    derivatives of ϕ are defined.
    """

    def hess(self, t, z, x=None):
        u"""Evaluate ϕ"(t) = dᵀ ∇²L(x + td, z) d.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        xtd = (self.x + t * self.d) if x is None else x
        return np.dot(self.d, self.model.hprod(xtd, z, self.d))

    def hprod(self, t, z, v, x=None):
        u"""Hessian-vector product, which is just ϕ"(t)*v with v ∈ ℝ.

        :keywords:
            :x: full-space x+td if that vector has already been formed.
        """
        return self.hess(t, z, x=x) * v
