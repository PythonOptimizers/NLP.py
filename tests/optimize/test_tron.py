import numpy as np
from nlp.tools.utils import breakpoints


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
