.. Description of ls and tr modules
.. _globalization-page:

************************
Globalization Techniques
************************

==================
Linesearch Methods
==================

The `linesearch` module in NLP.py implements a few typical line searches, i.e.,
given the current iterate :math:`x_k` along with the value :math:`f(x_k)`, the
gradient :math:`\nabla f(x_k)` and the search direction :math:`d_k`, they
return a stepsize :math:`\alpha_k > 0` such that :math:`x_k + \alpha_k d_k`
is an improved iterate in a certain sense. Typical conditions to be satisfied
by :math:`\alpha_k` include the ``Armijo condition``

.. math::

    f(x_k + \alpha_k d_k) \leq f(x_k) + \beta \alpha_k \nabla f(x_k)^T d_k

for some :math:`\beta \in (0,1)`, the ``Wolfe conditions``, which consist in
the Armijo condition supplemented with

.. math::

    \nabla f(x_k + \alpha_k d_k)^T d_k \geq \gamma \nabla f(x_k)^T d_k

for some :math:`\gamma \in (0,\beta)`, and the ``strong Wolfe conditions``,
which consist in the Armijo condition supplemented with

.. math::

    |\nabla f(x_k + \alpha_k d_k)^T d_k| \leq \gamma |\nabla f(x_k)^T d_k|

for some :math:`\gamma \in (0,\beta)`.

The :mod:`linesearch` Module
============================

.. _linesearch-section:

.. automodule:: linesearch

.. autoclass:: LineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: ArmijoLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`pyswolfe` Module
==========================

.. _pyswolfe-section:

.. automodule:: wolfe

.. autoclass:: StrongWolfeLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


The :mod:`pymswolfe` Module
===========================

.. _pymswolfe-section:

.. automodule:: pymswolfe

.. autoclass:: StrongWolfeLineSearch
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


====================
Trust-Region Methods
====================

Trust-region methods are an alternative to linesearch methods as a mechanism to
enforce global convergence, i.e., :math:`lim \nabla f(x_k) = 0`. In
trust-region methods, a quadratic model is approximately minimized at each
iteration subject to a trust-region constraint:

.. math::

    \begin{align*}
        \min_{d \in \mathbb{R}^n} & g_k^T d + \tfrac{1}{2} d^T H_k d \\
        \text{s.t.} & \|d\| \leq \Delta_k,
    \end{align*}

where :math:`g_k = \nabla f(x_k)`, :math:`H_k \approx \nabla^2 f(x_k)` and
:math:`\Delta_k > 0` is the current trust-region radius.

The :mod:`trustregion` Module
=============================

.. _trustregion-section:

.. automodule:: trustregion

.. autoclass:: TrustRegion
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: TrustRegionSolver
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:
