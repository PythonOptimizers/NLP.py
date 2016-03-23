"""Tests relative to PySparse."""

from helper import *
import numpy as np
from numpy.testing import *
import os

this_path = os.path.dirname(os.path.realpath(__file__))

if not module_missing('pysparse'):
    from nlp.model.pysparsemodel import PySparseAmplModel, PySparseSlackModel


class Test_PySparseAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    @dec.skipif(module_missing('pysparse'),
                "Test skipped because pysparse is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = PySparseAmplModel(model)    # x0 = (-1, ..., -1)


class Test_PySparseAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    @dec.skipif(module_missing('pysparse'),
                "Test skipped because pysparse is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_PySparseSlackHS7(TestCase, GenericTest):
    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    @dec.skipif(module_missing('pysparse'),
                "Test skipped because pysparse is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_PySparseSlackHS10(TestCase, GenericTest):
    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    @dec.skipif(module_missing('pysparse'),
                "Test skipped because pysparse is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs010.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1


if __name__ == '__main__':

    import unittest
    unittest.main()
