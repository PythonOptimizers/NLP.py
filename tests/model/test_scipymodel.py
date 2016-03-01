"""Tests relative to SciPy."""

from helper import *
import numpy as np
from numpy.testing import *
import os


this_path = os.path.dirname(os.path.realpath(__file__))

if not module_missing('scipy'):
    from nlp.model.scipymodel import SciPyAmplModel, SciPySlackModel


class Test_SciPyAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    @dec.skipif(module_missing('scipy'),
                "Test skipped because scipy is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = SciPyAmplModel(model)    # x0 = (-1, ..., -1)


class Test_SciPyAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    @dec.skipif(module_missing('scipy'),
                "Test skipped because scipy is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPyAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_SciPySlackHS7(TestCase, GenericTest):
    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    @dec.skipif(module_missing('scipy'),
                "Test skipped because scipy is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_SciPySlackHS10(TestCase, GenericTest):
    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    @dec.skipif(module_missing('scipy'),
                "Test skipped because scipy is not available.")
    def setUp(self):
        model = os.path.join(this_path, 'hs010.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1


if __name__ == '__main__':

    import unittest
    unittest.main()
