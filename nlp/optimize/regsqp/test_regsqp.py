"""Test damped LBFGS linear operators."""

from __future__ import division
import unittest
import pytest

import numpy as np
from numpy.testing import *

from new_regsqp import RegSQPSolver, FnormModel
import pysparse.sparse.pysparseMatrix as ps
from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.tools.dercheck import DerivativeChecker
from nlp.tools.logs import config_logger
import logging


class TestFnormModel(unittest.TestCase):
    """Test FnormModel."""

    def setUp(self):
        """Initialize."""
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        self.model = PySparseAmplModel('hs007.nl')
        self.fnormmodel = FnormModel(self.model)

    def test_derivatives(self):
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(self.fnormmodel,
                                   np.concatenate((self.model.x0,
                                                   np.ones(self.model.m))))
        dcheck.check(hess=False, chess=False)
        assert len(dcheck.grad_errs) == 0


class TestRegSQP(unittest.TestCase):
    """Test RegSQP solver."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        model = PySparseAmplModel('hs007.nl')
        self.solver = RegSQPSolver(model)
