.. Description of optimize module
.. _optimize-page:

****************
Complete Solvers
****************

.. _solvers-section:

==================
Linear Programming
==================

The linear programming problem can be stated in standard form as

.. math::

    \min_{x \in \mathbb{R}^n} \ c^T x \quad \text{subject to} \ Ax=b, \ x \geq
    0,

for some matrix :math:`A`. It is typical to reformulate arbitrary linear
programs as an equivalent linear program in standard form. However, in the next
solver, they are reformulated in so-called ``slack form`` using the
`SlackFramework` module. See :ref:`slacks-section`.

.. automodule:: lp

.. autoclass:: RegLPInteriorPointSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: RegLPInteriorPointSolver29
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

============================
Convex Quadratic Programming
============================

The convex quadratic programming problem can be stated in standard form as

.. math::

    \min_{x \in \mathbb{R}^n} \ c^T x + \tfrac{1}{2} x^T Q x \quad \text{subject to} \ Ax=b, \ x \geq
    0,

for some matrix :math:`A` and some square symmetric positive semi-definite
matrix :math:`Q`. It is typical to reformulate arbitrary quadratic
programs as an equivalent quadratic program in standard form. However, in the next
solver, they are reformulated in so-called ``slack form`` using the
`SlackFramework` module. See :ref:`slacks-section`.

.. automodule:: cqp

.. autoclass:: RegQPInteriorPointSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: RegQPInteriorPointSolver29
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

=========================
Unconstrained Programming
=========================

The unconstrained programming problem can be stated as

.. math::

    \min_{x \in \mathbb{R}^n} \ f(x)

for some smooth function :math:`f: \mathbb{R}^n \to \R`. Typically, :math:`f`
is required to be twice continuously differentiable.

The `trunk` solver requires access to exact first and second derivatives of
:math:`f`. If minimizes :math:`f` by solving a sequence of trust-region
subproblems, i.e., problems of the form

.. math::

    \min_{d \in \mathbb{R}^n} \ g_k^T d + \tfrac{1}{2} d^T H_k d \quad
    \text{s.t.} \ \|d\| \leq \Delta_k,

where :math:`g_k = \nabla f(x_k)`, :math:`H_k = \nabla^2 f(x_k)` and
:math:`\Delta_k > 0` is the current trust-region radius.

.. automodule:: trunk

.. autoclass:: Trunk
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: QNTrunk
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The L-BFGS method only requires access to exact first derivatives of
:math:`f` and maintains its own approximation to the second derivatives.

.. automodule:: lbfgs

.. autoclass:: LBFGS
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: WolfeLBFGS
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

=============================
Bound-Constrained Programming
=============================

The unconstrained programming problem can be stated as

.. math::

    \min_{x \in \R^n} \ f(x) \quad \text{s.t.} \ x_i \geq 0 \ (i \in
    \mathcal{B})

for some smooth function :math:`f: \R^n \to \R`. Typically, :math:`f`
is required to be twice continuously differentiable.

.. automodule:: tron

.. autoclass:: TRON
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

=============================
General Nonlinear Programming
=============================

The general nonlinear programming problem can be stated as

.. math::

    \begin{array}{ll}
      \min_{x \in \mathbb{R}^n} & f(x) \\
      \text{s.t.} & h(x) = 0, \\
                  & c(x) \geq 0,
    \end{array}

for smooth functions :math:`f`, :math:`h` and :math:`c`.

.. todo::

   Insert this module.
