import sys
try:
    from nlpy.model.scipymodel import SciPyAmplModel
except ImportError as exc:
    print "Failed to import: ", exc, "  No tests run!"
    sys.exit(0)

from helper import *
import numpy as np
import os


this_path = os.path.dirname(os.path.realpath(__file__))


class Test_SciPyAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, nlp):
        return get_derivatives_scipy(nlp)

    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.nlp = SciPyAmplModel(model)    # x0 = (-1, ..., -1)


class Test_SciPyAmplHS7(TestCase, Hs7):

    def get_derivatives(self, nlp):
        return get_derivatives_scipy(nlp)

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.nlp = SciPyAmplModel(model)
        self.nlp.pi0 = np.ones(1)


if __name__ == '__main__':

    import unittest
    unittest.main()
