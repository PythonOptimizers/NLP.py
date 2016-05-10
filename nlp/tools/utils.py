# -*- coding: utf-8 -*-
"""Utilities."""

import numpy as np
import logging
from math import copysign, sqrt
from nlp.tools.norms import norm2


def Max(a):
    """A safeguarded max function. Returns -infinity for empty arrays."""
    return np.max(a) if a.size > 0 else -np.inf


def Min(a):
    """A safeguarded min function. Returns +infinity for empty arrays."""
    return np.min(a) if a.size > 0 else np.inf


class NullHandler(logging.Handler):
    """A simple implementation of the null handler for Python 2.6.x.

    Useful for compatibility with older versions of Python.
    """

    def emit(self, record):
        pass

    def handle(self, record):
        pass

    def createLock(self):
        return None


# Helper functions.
def identical(a, b):
    """Check that two arrays or lists are identical.

    Must be cautious because of Numpy's strange behavior:
    >>> a = np.array([]) ; b = np.array([0])
    >>> np.all(a==b)
    True
    """
    if a.shape == b.shape:
        return np.all(a == b)
    return False


def where(cond):
    """Bypass Numpy's annoyances.

    Gee does someone need to write a proper Numpy!
    """
    return np.where(cond)[0]


def roots_quadratic(q2, q1, q0, tol=1.0e-8, nitref=1):
    """Find the real roots of the quadratic q(x) = q2 * x^2 + q1 * x + q0.

    The numbers q0, q1 and q0 must be real.

    This function takes after the GALAHAD function of the same name.
    See http://galahad.rl.ac.uk.
    """
    a2 = float(q2)
    a1 = float(q1)
    a0 = float(q0)

    # Case of a linear function.
    if a2 == 0.0:
        if a1 == 0.0:
            if a0 == 0.0:
                return [0.0]
            else:
                return []
        else:
            roots = [-a0 / a1]
    else:
        # Case of a quadratic.
        rhs = tol * a1 * a1
        if abs(a0 * a2) > rhs:
            rho = a1 * a1 - 4.0 * a2 * a0
            if rho < 0.0:
                return []
            # There are two real roots.
            d = -0.5 * (a1 + copysign(sqrt(rho), a1))
            roots = [d / a2, a0 / d]
        else:
            # Ill-conditioned quadratic.
            roots = [-a1 / a2, 0.0]

    # Perform a few Newton iterations to improve accuracy.
    new_roots = []
    for root in roots:
        for _ in xrange(nitref):
            val = (a2 * root + a1) * root + a0
            der = 2.0 * a2 * root + a1
            if der == 0.0:
                continue
            else:
                root = root - val / der
        new_roots.append(root)

    return new_roots


def to_boundary(x, p, delta, xx=None):
    u"""Compute a solution of the quadratic trust region equation.

    Return the largest (non-negative) solution of
        ‖x + σ p‖ = Δ.

    The code is only guaranteed to produce a non-negative solution
    if ‖x‖ ≤ Δ, and p != 0.
    If the trust region equation has no solution, σ is set to 0.

    :keywords:
        :xx: squared norm of argument `x`.
    """
    if delta is None:
        raise ValueError('`delta` value must be a positive number.')
    px = np.dot(p, x)
    pp = np.dot(p, p)
    if xx is None:
        xx = np.dot(x, x)
    d2 = delta**2

    # Guard against abnormal cases.
    rad = px**2 + pp * (d2 - xx)
    rad = sqrt(max(rad, 0.0))

    if px > 0:
        sigma = (d2 - xx) / (px + rad)
    elif rad > 0:
        sigma = (rad - px) / pp
    else:
        sigma = 0
    return sigma


def projected_gradient_norm2(x, g, l, u):
    """Compute the Euclidean norm of the projected gradient at x."""
    lower = where(x == l)
    upper = where(x == u)

    pg = g.copy()
    pg[lower] = np.minimum(g[lower], 0)
    pg[upper] = np.maximum(g[upper], 0)

    return norm2(pg[where(l != u)])


def project(x, l, u):
    """Project x into the box [l, u]."""
    return np.maximum(np.minimum(x, u), l)


def projected_step(x, d, l, u):
    """Project the step d into the box [l, u].

    The projected step is defined as s := P[x + d] - x.
    """
    return project(x + d, l, u) - x


def breakpoints(x, d, l, u):
    """Find the smallest and largest breakpoints on the half line x + t d.

    We assume that x is feasible. Return the smallest and largest t such
    that x + t d lies on the boundary.
    """
    pos = where((d > 0) & (x < u))  # Hit the upper bound.
    neg = where((d < 0) & (x > l))  # Hit the lower bound.
    npos = len(pos)
    nneg = len(neg)

    nbrpt = npos + nneg
    # Handle the exceptional case.
    if nbrpt == 0:
        return (0, 0, 0)

    brptmin = np.inf
    brptmax = 0
    if npos > 0:
        steps = (u[pos] - x[pos]) / d[pos]
        brptmin = min(brptmin, np.min(steps))
        brptmax = max(brptmax, np.max(steps))
    if nneg > 0:
        steps = (l[neg] - x[neg]) / d[neg]
        brptmin = min(brptmin, np.min(steps))
        brptmax = max(brptmax, np.max(steps))

    return (nbrpt, brptmin, brptmax)
