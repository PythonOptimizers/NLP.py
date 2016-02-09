import sys
try:
    from nlp.model.cysparsemodel import CySparseAmplModel
except ImportError as exc:
    print "Failed to import: ", exc, "  No tests run!"
    sys.exit(0)

from nlp.model.amplpy import AmplModel
from helper import *
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_CySparseAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = CySparseAmplModel(model)    # x0 = (-1, ..., -1)


class Test_CySparseAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = CySparseAmplModel(model)
        self.model.pi0 = np.ones(1)


if __name__ == '__main__':

    import unittest
    unittest.main()
