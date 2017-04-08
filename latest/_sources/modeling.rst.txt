.. Description of modeling module
.. _model-page:

==================
Modeling in NLP.py
==================

.. _nlp:

The general problem consists in minimizing an objective :math:`f(x)` subject to
general constraints and bounds:

.. math::

    c_i(x) = a_i,                 & \qquad i = 1, \ldots, m, \\
    g_j^L \leq g_j(x) \leq g_j^U, & \qquad j = 1, \ldots, p, \\
    x_k^L \leq x_k \leq x_k^U,    & \qquad k = 1, \ldots, n,

where some or all lower bounds :math:`g_j^L` and :math:`x_k^L` may be equal to
:math:`-\infty`, and some or all upper bounds :math:`g_j^U` and :math:`x_k^U`
may be equal to :math:`+\infty`.


.. _nlp-section:

--------------------------------
Modeling with `NLPModel` Objects
--------------------------------

`NLPModel` objects are the most basic type of model. They require that functions and their derivatives be coded explicitly.

The :mod:`nlp` Module
=====================

.. automodule:: nlpmodel

.. autoclass:: NLPModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. todo:: Insert example.


.. _amplpy-section:

------------------------------------
Models in the AMPL Modeling Language
------------------------------------

`AmplModel` objects are convenient in that the model is written in the AMPL modeling language, and the AMPL Solver Library takes care of evaluating derivatives for us.

See the `AMPL home page <http://www.ampl.com>`_ for more information on the
AMPL algebraic modeling language.

The :mod:`amplpy` Module
========================

.. automodule:: amplmodel

.. autoclass:: AmplModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. literalinclude:: ../../examples/demo_amplmodel.py
   :linenos:


--------------------------------------------
Python Models with Automatic Differentiation
--------------------------------------------

The :mod:`adolcmodel` Module
============================

.. automodule:: adolcmodel

.. autoclass:: AdolcModel
   :show-inheritance:
   :members:
   :undoc-members:


The :mod:`cppadmodel` Module
============================

.. automodule:: cppadmodel

.. autoclass:: CppADModel
  :show-inheritance:
  :members:
  :undoc-members:


The :mod:`algopymodel` Module
=============================

.. automodule:: algopymodel

.. autoclass:: AlgopyModel
 :show-inheritance:
 :members:
 :undoc-members:



--------------------
Reformulating Models
--------------------

.. _slacks-section:

Using the Slack Formulation of a Model
--------------------------------------

The :mod:`snlp` Module
========================

.. automodule:: snlp

.. autoclass:: SlackModel
   :show-inheritance:
   :members:
   :undoc-members:


Example
=======

.. todo:: Insert example.

Inheritance Diagrams
====================

.. inheritance-diagram:: nlp.model.nlpmodel nlp.model.amplmodel nlp.model.snlp nlp.model.adolcmodel nlp.model.cppadmodel nlp.model.algopymodel nlp.model.qnmodel nlp.model.linemodel nlp.model.pysparsemodel nlp.model.cysparsemodel nlp.model.scipymodel
  :parts: 1
