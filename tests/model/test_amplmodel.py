# Tests relative to algorithmic differentiation with AMPL.
from nlp.model.amplpy import AmplModel
from .helper import *
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_AmplRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock

    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = AmplModel(model)    # x0 = (-1, ..., -1)


class Test_AmplHS7(TestCase, Hs7):    # Test defined in Hs7

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AmplModel(model)    # x0 = (2, 2)
        self.model.pi0 = np.ones(1)


if __name__ == '__main__':

    import unittest
    unittest.main()
