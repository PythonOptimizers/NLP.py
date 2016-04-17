# -*- coding: utf-8 -*-
"""PDE-Constrained Optimization Problems.

A generic class to define a PDE-constrained optimization problem using FEniCS.
D. Orban, 2010--2016.
"""

# TODO: make `apply_bcs=True` everywhere?

from nlp.model.nlpmodel import NLPModel
import numpy as np
from dolfin import Function, TestFunction, TrialFunction, \
                   Constant, \
                   project, interpolate, \
                   assemble, derivative, action, adjoint, \
                   as_backend_type
import logging

__docformat__ = 'restructuredtext'


# Silence FFC.
ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)


class PDENLPModel(NLPModel):
    """PDE-constrained optimization problem.

    A generic class to represent PDE-constrained optimization problems
    modeled with Dolfin.
    """

    def __init__(self,
                 mesh,
                 function_space,
                 bcs,
                 name="Generic PDE model",
                 **kwargs):
        """Initialize a PDE-constrained optimization model.

        :parameters:
            :mesh: a mesh over the domain
            :function_space: a ``FunctionSpace`` instance
            :bcs: a list of boundary conditions
            :u0: an initial guess specified as an ``Expression`` or
                 ``Constant`` instance (default: ``u0 = Constant(0)``)
            :Lvar: lower bounds specified as an ``Expression`` or
                ``Constant`` instance (default: no lower bounds)
            :Uvar: upper bounds specified as an ``Expression`` or
                ``Constant`` instance (default: no upper bounds)
            :Lcon: constraint lhs specified as an ``Expression`` or
                ``Constant`` instance (default: no lower bounds)
            :Ucon: constraint rhs specified as an ``Expression`` or
                ``Constant`` instance (default: no upper bounds)

        :keywords:
            :name: a name given to the problems

        See ``NLPModel`` for other keyword arguments.
        """
        # Obtain initial guess and problem size.
        n = function_space.dim()
        self.__bcs = bcs

        # Set initial guess.
        self.__u0 = kwargs.pop("u0", None)
        if self.__u0 is None:
            nspaces = function_space.num_sub_spaces()

            # When function_space is not a mixed space, its number of subspaces
            # is zero...
            if nspaces > 0:
                self.__u0 = interpolate(Constant(tuple([0] * nspaces)),
                                        function_space)
            else:
                self.__u0 = interpolate(Constant(0), function_space)

        self.__u = Function(function_space)
        self.__u.assign(self.__u0)
        for bc in self.__bcs:
            bc.apply(self.__u.vector())

        # Set lower and upper bounds.
        self.__Lvar = kwargs.pop("Lvar", None)
        if self.__Lvar is not None:
            Lvar = assemble(self.__Lvar)
            for bc in self.__bcs:
                bc.apply(Lvar)
            kwargs["Lvar"] = Lvar.array()

        self.__Uvar = kwargs.pop("Uvar", None)
        if self.__Uvar is not None:
            Uvar = assemble(self.__Uvar)
            for bc in self.__bcs:
                bc.apply(Uvar)
            kwargs["Uvar"] = Uvar.array()

        # Set constraints lhs and rhs.
        ncon = 0
        self.__Lcon = kwargs.pop("Lcon", None)
        if self.__Lcon is not None:
            Lcon = assemble(self.__Lcon)
            for bc in self.__bcs:
                bc.apply(Lcon)
            kwargs["Lcon"] = Lcon.array()
            ncon = Lcon.array().size

        self.__Ucon = kwargs.pop("Ucon", None)
        if self.__Ucon is not None:
            Ucon = assemble(self.__Ucon)
            for bc in self.__bcs:
                bc.apply(Ucon)
            kwargs["Ucon"] = Ucon.array()

        self.__mesh = mesh
        self.__function_space = function_space

        # Remember function space where multipliers live.
        self.__dual_function_space = kwargs.get("dual_function_space", None)

        self._objective_functional = self.register_objective_functional()
        self.__objective_gradient = None
        self.__objective_hessian = None

        if self.__Lcon is not None or self.__Ucon is not None:
            self._constraint_form = self.register_constraint_form()
        else:
            self._constraint_form = None
        self.__constraint_jacobian = None
        self.__adjoint_jacobian = None
        self.__constraint_hessian = None

        # Initialize base class.
        super(PDENLPModel, self).__init__(n,
                                          m=ncon,
                                          name=name,
                                          x0=self.__u.vector().array(),
                                          **kwargs)

        return

    @property
    def mesh(self):
        """Return underlying mesh."""
        return self.__mesh

    @property
    def function_space(self):
        """Return underlying function space."""
        return self.__function_space

    @property
    def dual_function_space(self):
        """Return underlying dual function space."""
        return self.__dual_function_space

    @property
    def bcs(self):
        """Return problem boundary conditions."""
        return self.__bcs

    @property
    def u0(self):
        """Initial guess."""
        return self.__u0

    @property
    def u(self):
        """Current guess."""
        return self.__u

    def register_objective_functional(self):
        """Register a functional as objective function."""
        raise NotImplementedError("Must be subclassed.")

    def register_constraint_form(self):
        """Register a form as constraints."""
        raise NotImplementedError("Must be subclassed.")

    def assign_vector(self, x, apply_bcs=True):
        """Assign NumPy array or ``Expression`` ``x`` as current ``u``."""
        if isinstance(x, np.ndarray):
            self.u.vector()[:] = x
        else:
            xx = project(x, self.function_space)
            self.u.vector()[:] = xx.vector().array()

        # Apply boundary conditions if requested.
        apply_bcs = False
        if apply_bcs:
            for bc in self.__bcs:
                bc.apply(self.u.vector())

        return

    def obj(self, x, apply_bcs=True):
        """Evaluate the objective functional at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.
        """
        if self._objective_functional is None:
            msg = "Subclass and implement register_objective_functional()."
            raise NotImplementedError(msg)

        self.assign_vector(x, apply_bcs=apply_bcs)
        return assemble(self._objective_functional)

    def compile_objective_gradient(self):
        """Compute first variation of objective functional.

        This method only has side effects.
        """
        if self._objective_functional is None:
            raise NotImplementedError("Objective functional not registered.")

        # Fast return if first variation was already compiled.
        if self.__objective_gradient is not None:
            return

        du = TestFunction(self.function_space)
        self.__objective_gradient = derivative(self._objective_functional,
                                               self.u, du)
        return

    def grad(self, x, apply_bcs=True):
        """Evaluate the gradient of the objective functional at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          x = Expression('x[0] * sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        if self.__objective_gradient is None:
            self.compile_objective_gradient()

        self.assign_vector(x, apply_bcs=apply_bcs)
        g = assemble(self.__objective_gradient)
        return g.array()

    def compile_objective_hessian(self):
        """Compute second variation of objective functional.

        This method only has side effects.
        """
        # Fast return if second variation was already compiled.
        if self.__objective_hessian is not None:
            return

        # Make sure first variation was compiled.
        if self.__objective_gradient is None:
            self.compile_objective_gradient()

        du = TrialFunction(self.function_space)
        self.__objective_hessian = derivative(self.__objective_gradient,
                                              self.u, du)
        return

    def hess(self, x, y=None, apply_bcs=True, **kwargs):
        """Evaluate the Hessian matrix of the Lagrangian at (x,y).

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :y: Numpy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` and/or ``y`` are ``Expression``s, they must defined on the
        appropriate function spaces, by, e.g., declaring it as::

          Expression('x[0]*sin(x[1])',
                     element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        obj_weight = kwargs.get('obj_weight', 1.0)

        if self.__objective_hessian is None:
            self.compile_objective_hessian()

        self.assign_vector(x, apply_bcs=apply_bcs)
        H = obj_weight * assemble(self.__objective_hessian)

        if self.ncon > 0 and y is not None:
            if self.__constraint_jacobian is None:
                self.compile_constraint_jacobian()

            v = Function(self.__dual_function_space)
            v.vector()[:] = -y

            # ∑ⱼ yⱼ Hⱼ(u) may be viewed as the directional derivative
            # of J(.) with respect to u in the direction y.
            H += assemble(derivative(self.__constraint_jacobian, self.__u, v))

        return H

    def hprod(self, x, y, v, apply_bcs=True, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and v.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :y: Numpy ``array`` or Dolfin ``Expression`` for multipliers
            :v: Numpy ``array`` to multiply with
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          v = Expression('x[0]*sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        obj_weight = kwargs.get('obj_weight', 1.0)

        if self.__objective_hessian is None:
            self.compile_objective_hessian()

        self.assign_vector(x, apply_bcs=apply_bcs)

        w = Function(self.function_space)
        w.vector()[:] = v
        Hop = action(self.__objective_hessian, w)
        hv = obj_weight * assemble(Hop).array()

        if self.ncon > 0 and y is not None:
            if self.__constraint_jacobian is None:
                self.compile_constraint_jacobian()

            z = Function(self.__dual_function_space)
            z.vector()[:] = -y

            # ∑ⱼ yⱼ Hⱼ(u) may be viewed as the directional derivative
            # of J(.) with respect to u in the direction y.
            Hc = derivative(self.__constraint_jacobian, self.__u, z)
            hv += Hc * v

        return hv

    def cons(self, x, apply_bcs=True):
        """Evaluate the constraint form at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.
        """
        if self._constraint_form is None:
            msg = "Subclass and implement register_constraint_form()."
            raise NotImplementedError(msg)

        self.assign_vector(x, apply_bcs=apply_bcs)
        c = assemble(self._constraint_form)
        return c.array()

    def compile_constraint_jacobian(self):
        """Compute first variation of constraint form.

        This method only has side effects.
        """
        if self._constraint_form is None:
            raise NotImplementedError("Constraint form not registered.")

        # Fast return if first variation was already compiled.
        if self.__constraint_jacobian is not None:
            return

        du = TrialFunction(self.function_space)
        self.__constraint_jacobian = derivative(self._constraint_form,
                                                self.u, du)
        return

    def compile_adjoint_jacobian(self):
        """Compute the adjoint of the first variation of the constraint form.

        This method only has side effects.
        """
        self.compile_constraint_jacobian()
        self.__adjoint_jacobian = adjoint(self.__constraint_jacobian)
        return

    def jac(self, x, apply_bcs=True):
        """Evaluate the Jacobian of the constraints at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          x = Expression('x[0] * sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        if self.__constraint_jacobian is None:
            self.compile_constraint_jacobian()

        self.assign_vector(x, apply_bcs=apply_bcs)
        J = assemble(self.__constraint_jacobian)
        return J

    def jprod(self, x, v, apply_bcs=True, **kwargs):
        """Jacobian-vector product.

        Evaluate matrix-vector product between the constraint Jacobian at
        x and v.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :v: Numpy ``array`` to multiply with
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          v = Expression('x[0]*sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        if self.__constraint_jacobian is None:
            self.compile_constraint_jacobian()

        self.assign_vector(x, apply_bcs=apply_bcs)

        w = Function(self.function_space)
        w.vector()[:] = v
        Jop = action(self.__constraint_jacobian, w)
        jv = assemble(Jop)
        return jv.array()

    def jtprod(self, x, v, apply_bcs=True, **kwargs):
        """Transposed-Jacobian-vector product.

        Evaluate matrix-vector product between the transposed of the constraint
        Jacobian at x and v.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :v: Numpy ``array`` to multiply with
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          v = Expression('x[0]*sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        if self.__adjoint_jacobian is None:
            self.compile_adjoint_jacobian()

        self.assign_vector(x, apply_bcs=apply_bcs)

        w = Function(self.dual_function_space)
        w.vector()[:] = v
        JTop = action(self.__adjoint_jacobian, w)
        jtv = assemble(JTop)
        return jtv.array()


class DensePDENLPModel(PDENLPModel):
    """PDE-constrained model with derivatives as dense arrays."""

    def hess(self, *args, **kwargs):
        """Hessian as a dense array.

        See ``PDENPLModel.hess`` for more information.
        """
        H = super(DensePDENLPModel, self).hess(*args, **kwargs)
        return H.array()

    def jac(self, *args, **kwargs):
        """Jacobian as a dense array.

        See ``PDENPLModel.jac`` for more information.
        """
        J = super(DensePDENLPModel, self).jac(*args, **kwargs)
        return J.array()


class SparsePDENLPModel(PDENLPModel):
    """PDE-constrained model with derivatives in the backend format.

    How to obtain the nonzero elements of sparse matrices and their positions
    depends on the backend.
    """

    def hess(self, *args, **kwargs):
        """Hessian in the backend format.

        See ``PDENPLModel.hess`` for more information.
        """
        H = super(SparsePDENLPModel, self).hess(*args, **kwargs)
        return as_backend_type(H).mat()

    def jac(self, *args, **kwargs):
        """Jacobian in the backend format.

        See ``PDENPLModel.jac`` for more information.
        """
        J = super(SparsePDENLPModel, self).jac(*args, **kwargs)
        return as_backend_type(J).mat()


try:
    from scipy.sparse import csr_matrix

    class SciPyPDENLPModel(SparsePDENLPModel):
        """PDE-constrained model with derivatives in SciPy CSR format."""

        def hess(self, *args, **kwargs):
            """Hessian in SciPy CSR format.

            See ``PDENPLModel.hess`` for more information.
            """
            H = super(SciPyPDENLPModel, self).hess(*args, **kwargs)
            return csr_matrix(H.getValuesCSR()[::-1], shape=H.size)

        def jac(self, *args, **kwargs):
            """Jacobian in SciPy CSR format.

            See ``PDENPLModel.jac`` for more information.
            """
            J = super(SciPyPDENLPModel, self).jac(*args, **kwargs)
            return csr_matrix(J.getValuesCSR()[::-1], shape=J.size)

except:
    pass
