"""Tests relative to problems modeled with AMPL."""

from unittest import TestCase
from helper import *
import numpy as np
import os
import pytest

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_AmplRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock
    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = AmplModel(model)  # x0 = (-1, ..., -1)


class Test_AmplHS7(TestCase, Hs7):  # Test defined in Hs7
    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AmplModel(model)  # x0 = (2, 2)
        self.model.pi0 = np.ones(1)


class Test_PySparseAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = PySparseAmplModel(model)    # x0 = (-1, ..., -1)


class Test_PySparseAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_PySparseSlackHS7(TestCase, GenericTest):
    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_PySparseSlackHS10(TestCase, GenericTest):
    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1


class Test_SciPyAmplRosenbrock(TestCase, Rosenbrock):
    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = SciPyAmplModel(model)  # x0 = (-1, ..., -1)


class Test_SciPyAmplHS7(TestCase, Hs7):
    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPyAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_SciPySlackHS7(TestCase, GenericTest):
    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_SciPySlackHS10(TestCase, GenericTest):
    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1
