"""Tests relative to TRON."""

from helper import *
from nlp.optimize.tron import TRON

import numpy as np
import pytest
import os

this_path = os.path.dirname(os.path.realpath(__file__))


def test_breakpoints():
    """Test :meth: breakpoints from :class: TRON."""
    pytest.importorskip("nlp.model.amplpy")
    model = os.path.join(this_path, "..", "model", "rosenbrock.nl")
    return TRON(AmplModel(model))

    xl = np.array([-3, -2])
    xu = np.array([2, 1])

    # x is strictly inside boundaries
    x = np.array([-2, -1])
    d = np.array([-1, 1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 2)
    assert(brptmin == 1)
    assert(brptmax == 2)

    d = np.array([2, 1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 2)
    assert(brptmin == 2)
    assert(brptmax == 2)

    d = np.array([2, -1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 2)
    assert(brptmin == 1)
    assert(brptmax == 2)

    d = np.array([-1, -1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 2)
    assert(brptmin == 1)
    assert(brptmax == 1)

    d = np.array([0, 0])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 0)
    assert(brptmin == 0)
    assert(brptmax == 0)

    # x is on both boundaries
    x = np.array([2, 1])
    d = np.array([-1, -1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 2)
    assert(brptmin == 3)
    assert(brptmax == 5)

    d = np.array([1, 1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 0)
    assert(brptmin == 0)
    assert(brptmax == 0)

    d = np.array([1, 1])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 0)
    assert(brptmin == 0)
    assert(brptmax == 0)

    # x is on one boundary
    x = np.array([-1, 1])
    d = np.array([1, 2])
    (nbrpt, brptmin, brptmax) = self.tron.breakpoints(x, d, xl, xu)
    assert(nbrpt == 1)
    assert(brptmin == 3)
    assert(brptmax == 3)
