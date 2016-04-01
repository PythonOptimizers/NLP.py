from nlp.model.nlpmodel import NLPModel, UnconstrainedNLPModel
from nlp.tools.dercheck import DerivativeChecker
import numpy as np
import pytest


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

    def hess(self, x, *args, **kwargs):
        n = self.nvar
        d = np.empty(n)
        d[:-1] = 800 * x[:-1]**2 - 400 * (x[1:] - x[:-1]**2) + 2
        d[1:] += 200
        d[-1] = 200
        o = -400 * x[:-1]
        return np.diag(d) + np.diag(o, 1) + np.diag(o, -1)


@pytest.fixture(params=[2, 5, 10])
def rosenbrock_checker(request):
    model = Rosenbrock(request.param)
    x = np.ones(model.nvar)
    x[1::2] = -1
    dcheck = DerivativeChecker(model, x, tol=1.0e-4)
    dcheck.check()
    dcheck.check(cheap_check=True, hess=False)
    return dcheck


def test_rosenbrock(rosenbrock_checker):
    assert (len(rosenbrock_checker.grad_errs) == 0)
    assert (len(rosenbrock_checker.hess_errs) == 0)
    assert (len(rosenbrock_checker.cheap_grad_errs) == 0)


class Erroneous(NLPModel):

    def __init__(self, nvar, **kwargs):
        kwargs.pop("m", None)
        super(Erroneous, self).__init__(nvar, m=1, **kwargs)

    def obj(self, x):
        return 0.5 * np.dot(x, x)

    def grad(self, x):
        g = x.copy()
        g[0] += 1  # error: should be x[0].
        return g

    def hess(self, x, *args, **kwargs):
        obj_weight = kwargs.get('obj_weight', 1.0)
        H = np.eye(self.nvar)
        # H[0, 0] will appear correct by accident.
        H[1, 1] = 2  # error: should be 1.
        H[1, 2] = H[2, 1] = 2  # error: should be 0.
        return obj_weight * H

    def cons(self, x):
        return np.array([2 * np.sum(x) + 3])

    def jac(self, x):
        J = np.empty((self.ncon, self.nvar))
        J[0, :] = 2 * np.ones_like(x)
        J[0, -1] -= 3.14  # error: should be 2.
        return J

    def igrad(self, i, x):
        if i != 0:
            raise IndexError("Invalid constraint index")
        J = self.jac(x)
        return J[0, :].copy()


@pytest.fixture(params=[5, 10])
def erroneous_checker(request):
    model = Erroneous(request.param)
    x = np.random.random(model.nvar)
    dcheck = DerivativeChecker(model, x, tol=1.0e-5)
    dcheck.check()
    dcheck.check(cheap_check=True, hess=False, jac=False, chess=False)
    return dcheck


def test_erroneous(erroneous_checker):
    assert (len(erroneous_checker.grad_errs) == 1)
    assert (0 in erroneous_checker.grad_errs)
    assert (np.allclose(erroneous_checker.grad_errs[0], 1))

    assert (len(erroneous_checker.cheap_grad_errs) == 2)

    assert (len(erroneous_checker.hess_errs) == 2)
    assert ((1, 1) in erroneous_checker.hess_errs)
    assert (np.allclose(erroneous_checker.hess_errs[(1, 1)], 1))
    assert ((2, 1) in erroneous_checker.hess_errs)
    assert (np.allclose(erroneous_checker.hess_errs[(2, 1)], 2))

    n = erroneous_checker.model.nvar
    assert (len(erroneous_checker.jac_errs) == 1)
    assert ((0, n-1) in erroneous_checker.jac_errs)
    assert (np.allclose(erroneous_checker.jac_errs[(0, n-1)], 3.14 / 2))

    m = erroneous_checker.model.ncon
    for j in xrange(m):
        assert (len(erroneous_checker.chess_errs[j]) == 0)
