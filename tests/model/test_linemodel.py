# -*- coding: utf-8 -*-
from nlp.model.nlpmodel import NLPModel, UnconstrainedNLPModel, \
                               BoundConstrainedNLPModel
from nlp.model.linemodel import C1LineModel, C2LineModel
from nlp.tools.utils import where
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


@pytest.fixture(params=[2, 5, 10])
def c1rosenbrock_restriction(request):
    return C1LineModel(Rosenbrock(request.param),
                       np.zeros(request.param),
                       np.ones(request.param))


def test_c1rosenbrock(c1rosenbrock_restriction):
    model = c1rosenbrock_restriction.model
    x = np.zeros(model.nvar)
    d = np.ones(model.nvar)
    assert (c1rosenbrock_restriction.nvar == 1)
    assert (c1rosenbrock_restriction.ncon == 0)
    assert np.allclose(c1rosenbrock_restriction.obj(0), model.obj(x))
    assert np.allclose(c1rosenbrock_restriction.obj(1), model.obj(x + d))

    assert np.allclose(c1rosenbrock_restriction.grad(0),
                       np.dot(model.grad(x), d))
    assert np.allclose(c1rosenbrock_restriction.grad(1),
                       np.dot(model.grad(x + d), d))

    with pytest.raises(NotImplementedError):
        c1rosenbrock_restriction.hess(0)

    with pytest.raises(NotImplementedError):
        c1rosenbrock_restriction.hprod(0, 0, 2)

@pytest.fixture(params=[2, 5, 10])
def c2rosenbrock_restriction(request):
    return C2LineModel(Rosenbrock(request.param),
                       np.zeros(request.param),
                       np.ones(request.param))


def test_c2rosenbrock(c2rosenbrock_restriction):
    model = c2rosenbrock_restriction.model
    x = np.zeros(model.nvar)
    d = np.ones(model.nvar)
    assert (c2rosenbrock_restriction.nvar == 1)
    assert (c2rosenbrock_restriction.ncon == 0)
    assert np.allclose(c2rosenbrock_restriction.obj(0), model.obj(x))
    assert np.allclose(c2rosenbrock_restriction.obj(1), model.obj(x + d))

    assert np.allclose(c2rosenbrock_restriction.grad(0),
                       np.dot(model.grad(x), d))
    assert np.allclose(c2rosenbrock_restriction.grad(1),
                       np.dot(model.grad(x + d), d))

    assert np.allclose(c2rosenbrock_restriction.hess(0, 0),
                       np.dot(d, model.hprod(x, 0, d)))
    assert np.allclose(c2rosenbrock_restriction.hess(1, 0),
                       np.dot(d, model.hprod(x + d, 0, d)))

    assert np.allclose(c2rosenbrock_restriction.hprod(0, 0, 2),
                       2 * np.dot(d, model.hprod(x, 0, d)))


class BoundedRosenbrock(BoundConstrainedNLPModel):
    def __init__(self, nvar, Lvar, Uvar, **kwargs):
        assert (nvar > 1)
        super(BoundedRosenbrock, self).__init__(nvar, Lvar, Uvar, **kwargs)

    def obj(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


@pytest.fixture(params=[2, 5, 10])
def c1boundedrosenbrock_restriction_feas(request):
    nvar = request.param
    # The original model has bounds 0 ≤ x ≤ 1.
    # We choose an x inside the bounds and a random d.
    return C1LineModel(BoundedRosenbrock(nvar, np.zeros(nvar), np.ones(nvar)),
                       np.random.random(nvar),
                       np.random.random(nvar) - 0.5)


def test_bounded_rosenbrock_feas(c1boundedrosenbrock_restriction_feas):
    linemodel = c1boundedrosenbrock_restriction_feas
    model = linemodel.model
    x = linemodel.x
    d = linemodel.d
    tmin = linemodel.Lvar[0]
    tmax = linemodel.Uvar[0]
    assert tmin <= tmax
    assert np.all(model.Lvar <= x + tmin * d)
    assert np.all(x + tmin * d <= model.Uvar)
    assert np.all(model.Lvar <= x + tmax * d)
    assert np.all(x + tmax * d <= model.Uvar)


@pytest.fixture(params=[2, 5, 10])
def c1boundedrosenbrock_restriction_infeas(request):
    nvar = request.param
    # The original model has bounds 0 ≤ x ≤ 1.
    # We choose an x outside the bounds and  d.
    x = np.zeros(nvar)
    x[0] = 2
    return C1LineModel(BoundedRosenbrock(nvar, np.zeros(nvar), np.ones(nvar)),
                       x,
                       np.ones(nvar))


def test_bounded_rosenbrock_infeas(c1boundedrosenbrock_restriction_infeas):
    linemodel = c1boundedrosenbrock_restriction_infeas
    tmin = linemodel.Lvar[0]
    tmax = linemodel.Uvar[0]
    assert tmin > tmax
