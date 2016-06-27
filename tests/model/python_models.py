from nlp.model.nlpmodel import NLPModel, UnconstrainedNLPModel
import numpy as np


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


class SimpleQP(NLPModel):

    def __init__(self, **kwargs):
        super(SimpleQP, self).__init__(
            n=2, m=1, Lcon=np.zeros(1), Ucon=np.zeros(1), **kwargs)

    def obj(self, x):
        return .5 * (x[0]**2 + x[1]**2)

    def grad(self, x):
        n = self.nvar
        g = np.empty(n)
        g[0] = x[0]
        g[1] = x[1]
        return g

    def cons(self, x):
        c = np.empty(self.m)
        c[0] = x[1]**3 + 1
        return c

    def jac(self, x):
        J = np.empty([1, self.nvar])
        J[0, 0] = 0
        J[0, 1] = 3 * x[1]**2
        return J

    def jprod(self, x, v):
        return np.dot(self.jac(x), v)

    def jtprod(self, x, v):
        return np.dot(self.jac(x).T, v)

    def hess(self, x, z=None, *args, **kwargs):

        if z is None:
            z = np.zeros(self.m)
        H = np.empty([self.n, self.n])
        H[0, 0] = 1
        H[0, 1] = H[1, 0] = 0
        H[1, 1] = 1 - 6 * z * x[1]
        return H

    def hprod(self, x, z, v):
        hv = np.empty(self.n)
        hv[0] = v[0]
        hv[1] = (1 - 6 * z * x[1]) * v[1]
        return hv


class SimpleCubicProb(UnconstrainedNLPModel):

    def __init__(self, **kwargs):
        super(SimpleCubicProb, self).__init__(2, **kwargs)

    def obj(self, x):
        return x[0]**3 + x[1]**3

    def grad(self, x):
        n = self.nvar
        g = np.empty(n)
        g[0] = 3 * x[0]**2
        g[1] = 3 * x[1]**2
        return g

    def hess(self, x, z=None, *args, **kwargs):

        if z is None:
            z = np.zeros(self.m)
        H = np.empty([self.n, self.n])
        H[0, 0] = 6. * x[0]
        H[0, 1] = H[1, 0] = 0
        H[1, 1] = 6. * x[1]
        return H


class LineSearchProblem(UnconstrainedNLPModel):

    def __init__(self, **kwargs):
        super(LineSearchProblem, self).__init__(1, **kwargs)

    def obj(self, x):
        return (x - 3) * x**3 * (x - 6)**4

    def grad(self, x):
        g = (-6 + x)**3 * x**2 * (54 - 45 * x + 8 * x**2)
        return g

    def hess(self, x, z=None, *args, **kwargs):
        H = 2 * (-6 + x)**2 * x * (-324 + 540 * x - 231 * x**2 + 28 * x**3)
        return H


class LineSearchProblem2(UnconstrainedNLPModel):

    def __init__(self, **kwargs):
        super(LineSearchProblem2, self).__init__(1, **kwargs)

    def obj(self, x):
        return (x**2 - 1) * x**2 * (x**2 - 16) + 200

    def grad(self, x):
        g = 32 * x - 68 * x**3 + 6 * x**5
        return g

    def hess(self, x, z=None, *args, **kwargs):
        H = 32 - 204 * x**2 + 30 * x**4
        return H
