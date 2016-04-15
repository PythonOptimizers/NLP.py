"""Tests relative to CySparse."""

import os
import pytest
from unittest import TestCase
from helper import *
import numpy as np


this_path = os.path.dirname(os.path.realpath(__file__))


class Test_CySparseAmplRosenbrock(TestCase, Rosenbrock):
    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = CySparseAmplModel(model)  # x0 = (-1, ..., -1)


class Test_CySparseAmplHS7(TestCase, Hs7):
    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = CySparseAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_CySparseSlackHS7(TestCase, GenericTest):
    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = CySparseSlackModel(CySparseAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_CySparseSlackHS10(TestCase, GenericTest):
    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("cysparse")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = CySparseSlackModel(CySparseAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1
