import numpy as np
from nlp.tools.utils import *


def test_breakpoints():
    xl = np.array([-3, -2])
    xu = np.array([2, 1])

    # x is strictly inside the bounds
    x = np.array([-2, -1])
    d = np.array([-1, 1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 2
    assert brptmin == 1
    assert brptmax == 2

    d = np.array([2, 1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 2
    assert brptmin == 2
    assert brptmax == 2

    d = np.array([2, -1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 2
    assert brptmin == 1
    assert brptmax == 2

    d = np.array([-1, -1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 2
    assert brptmin == 1
    assert brptmax == 1

    d = np.array([0, 0])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 0
    assert brptmin == 0
    assert brptmax == 0

    # two bounds are active at x
    x = np.array([2, 1])
    d = np.array([-1, -1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 2
    assert brptmin == 3
    assert brptmax == 5

    d = np.array([1, 1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 0
    assert brptmin == 0
    assert brptmax == 0

    d = np.array([1, 1])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 0
    assert brptmin == 0
    assert brptmax == 0

    # one bound is active at x
    x = np.array([-1, 1])
    d = np.array([1, 2])
    (nbrpt, brptmin, brptmax) = breakpoints(x, d, xl, xu)
    assert nbrpt == 1
    assert brptmin == 3
    assert brptmax == 3


def test_to_boundary():

    x = np.array([2.])
    p = np.array([1.])
    delta = 2

    assert to_boundary(x, p, delta) == 0
    assert to_boundary(x, p, delta, 4.) == 0

    x = np.array([1., 1.])
    p = np.array([4., 4.])
    delta = np.sqrt(8)

    np.testing.assert_approx_equal(to_boundary(x, p, delta), 0.25)
    np.testing.assert_approx_equal(to_boundary(x, p, delta, 2.), 0.25)


def test_roots_quadratic():
    roots = roots_quadratic(1., 2., 1., tol=1.0e-8, nitref=1)
    assert roots == [-1., -1.]

    roots = roots_quadratic(1., 0, -2., tol=1.0e-8, nitref=1)
    np.testing.assert_approx_equal(roots[0], -np.sqrt(2))
    np.testing.assert_approx_equal(roots[1], np.sqrt(2))
