# -*- coding: utf-8 -*-
"""Damped Limited-Memory BFGS Operators.

Linear operators to represent limited-memory BFGS matrices and their inverses
using Powell damped update.
"""

from pykrylov.linop.lbfgs import InverseLBFGSOperator, LBFGSOperator
import numpy as np

__docformat__ = 'restructuredtext'


class DampedInverseLBFGSOperator(InverseLBFGSOperator):
    """Inverse LBFGS operator with damping."""

    def __init__(self, *args, **kwargs):
        u"""Instantiate a :class: `DampedInverseLBFGSOperator`.

        The arguments are the same as `InverseLBFGSOperator`.
        """
        super(DampedInverseLBFGSOperator, self).__init__(*args, **kwargs)
        self.eta = 0.2

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}."""
        ys = np.dot(new_s, new_y)
        By = self.qn_matvec(new_y)
        yBy = np.dot(new_y, By)
        s = new_s
        if ys < self.eta * yBy:
            theta = (1 - self.eta) * yBy / (yBy - ys)
            s = theta * new_s + (1 - theta) * By
            ys = self.eta * yBy

        if ys > self.accept_threshold:
            insert = self.insert
            self.s[:, insert] = s.copy()
            self.y[:, insert] = new_y.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        return


class DampedLBFGSOperator(LBFGSOperator):
    """LBFGS operator with damping."""

    def __init__(self, *args, **kwargs):
        u"""Instantiate a :class: `DampedLBFGSOperator`.

        The arguments are the same as `LBFGSOperator`.
        """
        super(DampedLBFGSOperator, self).__init__(*args, **kwargs)
        self.eta = 0.2

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}."""
        ys = np.dot(new_s, new_y)
        Bs = self.qn_matvec(new_s)
        sBs = np.dot(new_s, Bs)
        y = new_y
        if ys < self.eta * sBs:
            theta = (1 - self.eta) * sBs / (sBs - ys)
            y = theta * new_y + (1 - theta) * Bs
            ys = self.eta * sBs

        if ys > self.accept_threshold:
            insert = self.insert
            self.s[:, insert] = new_s.copy()
            self.y[:, insert] = y.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        return
