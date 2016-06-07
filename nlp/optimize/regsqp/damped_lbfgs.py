# -*- coding: utf-8 -*-
"""Damped Limited-Memory BFGS Operators.

Linear operators to represent limited-memory BFGS matrices and their inverses
using Powell damped update.
"""

from pykrylov.linop.lbfgs import InverseLBFGSOperator, LBFGSOperator
import numpy as np

__docformat__ = 'restructuredtext'


class DampedInverseLBFGSOperator(InverseLBFGSOperator):
    """

    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(DampedInverseLBFGSOperator, self).__init__(n, npairs, **kwargs)
        self.eta = 0.2

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = np.dot(new_s, new_y)
        By = self.qn_matvec(new_y)
        yBy = np.dot(new_y, By)
        # print 'yBy:', yBy
        # if np.dot(new_y, new_s) >= self.eta * yBy:
        if ys >= self.eta * yBy:
            theta = 1.0
        else:
            theta = (1 - self.eta) * yBy / (yBy - ys)

        # print 'theta:', theta
        s = theta * new_s + (1 - theta) * By
        ys = theta * ys + (1 - theta) * yBy
        # ys = np.dot(s, new_y)
        # print 'ys:', ys

        insert = self.insert
        self.s[:, insert] = s.copy()
        self.y[:, insert] = new_y.copy()
        self.ys[insert] = ys
        self.insert += 1
        self.insert = self.insert % self.npairs
        return


class DampedLBFGSOperator(LBFGSOperator):
    """

    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(DampedLBFGSOperator, self).__init__(n, npairs, **kwargs)
        self.eta = 0.2

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = np.dot(new_s, new_y)
        Bs = self.qn_matvec(new_s)
        sBs = np.dot(new_s, Bs)
        # print "sBs: ", sBs
        # if np.dot(new_y, new_s) >= self.eta * sBs:
        if ys >= self.eta * sBs:
            theta = 1.0
        else:
            theta = (1 - self.eta) * sBs / (sBs - ys)

        # print 'theta: ', theta

        y = theta * new_y + (1 - theta) * Bs
        ys = theta * ys + (1 - theta) * sBs
        # ys = np.dot(new_s, y)

        # print 'ys:', ys

        insert = self.insert
        self.s[:, insert] = new_s.copy()
        self.y[:, insert] = y.copy()
        self.ys[insert] = ys
        self.insert += 1
        self.insert = self.insert % self.npairs
        return
