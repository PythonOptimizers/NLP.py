"""Tests relative to problems modeled with AMPL."""

from unittest import TestCase
from nlp.model.amplpy import AmplModel
from helper import *
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_AmplRosenbrock(TestCase, Rosenbrock):  # Test def'd in Rosenbrock
    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = AmplModel(model)  # x0 = (-1, ..., -1)


class Test_AmplHS7(TestCase, Hs7):  # Test defined in Hs7
    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AmplModel(model)  # x0 = (2, 2)
        self.model.pi0 = np.ones(1)
