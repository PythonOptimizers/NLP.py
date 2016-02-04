import sys
try:
    from nlpy.model.pysparsemodel import PySparseAmplModel
except ImportError as exc:
    print "Failed to import: ", exc, "  No tests run!"
    sys.exit(0)

from helper import *
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_PySparseAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, nlp):
        return get_derivatives_llmat(nlp)

    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.nlp = PySparseAmplModel(model)    # x0 = (-1, ..., -1)


class Test_PySparseAmplHS7(TestCase, Hs7):

    def get_derivatives(self, nlp):
        return get_derivatives_llmat(nlp)

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.nlp = PySparseAmplModel(model)
        self.nlp.pi0 = np.ones(1)


if __name__ == '__main__':

    import unittest
    unittest.main()
