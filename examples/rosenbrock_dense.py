"""A simple implementation of the generalized Rosenbrock problem.

The Hessian is returned as a dense Numpy array.
"""

import numpy as np
from nlp.model.nlpmodel import UnconstrainedNLPModel
from nlp.model.qnmodel import QuasiNewtonModel


class Rosenbrock(UnconstrainedNLPModel):
    def __init__(self, nvar, **kwargs):
        assert (nvar > 1)
        super(Rosenbrock, self).__init__(nvar, **kwargs)

    def obj(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def grad(self, x):
        n = self.nvar
        g = np.empty(n)
        g[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        g[-1] = 200 * (x[-1] - x[-2]**2)
        g[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - \
                  400 * x[1:-1] * (x[2:] - x[1:-1]**2) - 2 * (1 - x[1:-1])
        return g

    def diags(self, x):
        n = self.nvar
        d = np.empty(n)
        d[:-1] = 800 * x[:-1]**2 - 400 * (x[1:] - x[:-1]**2) + 2
        d[1:] += 200
        d[-1] = 200
        o = -400 * x[:-1]
        return (d, o)

    def hess(self, x, z):
        (d, o) = self.diags(x)
        return np.diag(d) + np.diag(o, 1) + np.diag(o, -1)

    def hprod(self, x, z, v):
        (d, o) = self.diags(x)
        hv = d * v
        hv[1:] += o * x[1:]
        hv[:-1] += o * x[:-1]
        return hv


class QNRosenbrock(QuasiNewtonModel, Rosenbrock):
    """Rosenbrock problem with quasi-Newton Hessian approximation.

    Initialize using::

        model = QNRosenbrock(5, H=InverseLBFGSOperator)
    """

    pass  # All the work is done by the parent classes.
