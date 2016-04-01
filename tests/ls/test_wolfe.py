from nlp.model.nlpmodel import UnconstrainedNLPModel
from nlp.model.linemodel import C1LineModel
from nlp.ls.wolfe import StrongWolfeLineSearch

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


@pytest.fixture(params=[2, 5, 10])
def rosenbrock_wolfe(request):
    model = Rosenbrock(request.param)
    x = np.zeros(request.param)
    g = model.grad(x)
    c1model = C1LineModel(model, x, -g)  # steepest descent direction
    return StrongWolfeLineSearch(c1model)


def test_c1rosenbrock(rosenbrock_wolfe):
    with pytest.raises(StopIteration):
        while True:
            rosenbrock_wolfe.next()


@pytest.fixture(params=[2, 5, 10])
def rosenbrock_wolfe_ascent(request):
    model = Rosenbrock(request.param)
    x = np.zeros(request.param)
    g = model.grad(x)
    c1model = C1LineModel(model, x, g)  # ascent direction!
    return c1model


def test_c1rosenbrock_ascent(rosenbrock_wolfe_ascent):
    with pytest.raises(ValueError):
        StrongWolfeLineSearch(rosenbrock_wolfe_ascent)
