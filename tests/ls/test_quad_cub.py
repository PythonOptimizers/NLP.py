from nlp.model.linemodel import C1LineModel
from nlp.ls.quad_cub import QuadraticCubicLineSearch

from unittest import TestCase
import numpy as np
import pytest
import os
import sys
this_path = os.path.dirname(os.path.realpath(__file__))

lib_path = os.path.abspath(os.path.join(this_path, '..', 'model'))
sys.path.append(lib_path)

from python_models import SimpleQP, SimpleCubicProb, LineSearchProblem2


class Test_SimpleQPQuadCubLS(TestCase):

    def setUp(self):
        # pytest.importorskip("nlp.model.amplmodel")
        # model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = SimpleQP()  # x0 = (-1, ..., -1)

    def test_ascent(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, g)
        with pytest.raises(ValueError):
            QuadraticCubicLineSearch(c1model)

    def test_ls_steepest(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, -g)
        ls = QuadraticCubicLineSearch(c1model)
        with pytest.raises(StopIteration):
            ls.next()
        np.allclose(ls.iterate, np.zeros(self.model.n))

    def test_ls_2steepest(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, -g)
        ls = QuadraticCubicLineSearch(c1model, step=2.)
        ls.next()
        with pytest.raises(StopIteration):
            ls.next()
        np.allclose(ls.iterate, np.zeros(self.model.n))


class Test_SimpleCubicProbQuadCubLS(TestCase):

    def setUp(self):
        self.model = SimpleCubicProb()

    def test_ascent(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, g)
        with pytest.raises(ValueError):
            QuadraticCubicLineSearch(c1model)

    def test_ls_steepest(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, -g)
        ls = QuadraticCubicLineSearch(c1model)
        with pytest.raises(StopIteration):
            ls.next()
        np.allclose(ls.iterate, np.zeros(self.model.n))


class Test_LineSearchProblem2QuadCubLS(TestCase):

    def setUp(self):
        self.model = LineSearchProblem2()

    def test_ascent(self):
        x = np.ones(self.model.n)
        g = self.model.grad(x)
        c1model = C1LineModel(self.model, x, g)
        with pytest.raises(ValueError):
            QuadraticCubicLineSearch(c1model)

    def test_ls(self):
        x = np.array([-4.])
        g = np.array([1])
        c1model = C1LineModel(self.model, x, g)
        ls = QuadraticCubicLineSearch(c1model, step=8)
        ls.next()
        ls.next()
        with pytest.raises(StopIteration):
            ls.next()
        np.allclose(ls.iterate, np.array([1.69059892324]))
