# -*- coding: utf8 -*-

"""A simple derivative checker."""

import numpy as np
from numpy.linalg import norm
from math import sqrt
import logging

np.random.seed(0)
macheps = np.finfo(np.double).eps  # Machine epsilon.


class DerivativeChecker(object):
    """Verify numerically the accuracy of first and second derivatives.

    The `DerivativeChecker` class provides facilities for verifying
    numerically the accuracy of first and second-order derivatives
    implemented in an optimization model.
    """

    def __init__(self, model, x, **kwargs):
        u"""Initialize a :class:`DerivativeChecker` instance.

        :parameters:
            :model: a `NLPModel` instance
            :x:     the point about which we are checking derivatives.
                    See the documentation of :meth:`check` for options.

        :keywords:
            :tol:         tolerance under which derivatives are considered
                            accurate (default: 100 * √ϵ)
            :step:        centered finite difference step, will be scaled
                            by (1 + ‖x‖₁) (default: ³√(ϵ/3))
            :logger_name: name of a logger object (default: None)
        """
        self.tol = kwargs.get('tol', 100 * sqrt(macheps))
        self.step = kwargs.get('step', (macheps / 3)**(1. / 3))
        self.h = self.step * (1 + norm(x, 1))

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlp.der')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False

        self.model = model
        self.x = x.copy()
        self.grad_errs = {}
        self.cheap_grad_errs = {}
        self.jac_errs = {}
        self.hess_errs = {}
        self.chess_errs = {}

        headfmt = '%4s  %4s        %22s  %22s  %7s'
        self.head = headfmt % ('Fun', 'Var', 'Expected',
                               'Finite Diff', 'Rel.Err')
        self.d1fmt = '%4d  %4d        %22.15e  %22.15e  %7.1e'
        head2fmt = '%4s  %4s  %4s  %22s  %22s  %7s'
        self.head2 = head2fmt % ('Fun', 'Var', 'Var', 'Expected',
                                 'Finite Diff', 'Rel.Err')
        self.d2fmt = '%4d  %4d  %4d  %22.15e  %22.15e  %7.1e'
        head3fmt = '%17s %22s  %22s  %7s'
        self.head3 = head3fmt % ('Directional Deriv', 'Expected',
                                 'Finite Diff', 'Rel.Err')
        self.d3fmt = '%17s %22.15e  %22.15e  %7.1e'

        return

    def check(self, **kwargs):
        """Perform derivative check.

        :keywords:
            :grad:      Check objective gradient  (default `True`)
            :hess:      Check objective Hessian   (default `True`)
            :jac:       Check constraints Jacobian (default `True` if m > 0)
            :chess:     Check constraints Hessian (default `True` if m > 0)
        """
        grad = kwargs.get('grad', True)
        hess = kwargs.get('hess', True)
        jac = kwargs.get('jac', True) if self.model.ncon > 0 else False
        chess = kwargs.get('chess', True) if self.model.ncon > 0 else False
        cheap = kwargs.get('cheap_check', False)

        self.log.debug('Gradient checking')

        if grad:
            if cheap:
                self.cheap_grad_errs = self.cheap_check_obj_gradient()
            else:
                self.grad_errs = self.check_obj_gradient()
        if jac:
            self.jac_errs = self.check_con_jacobian()
        if hess:
            self.hess_errs = self.check_obj_hessian()
        if chess:
            self.chess_errs = self.check_con_hessians()

        return

    def cheap_check_obj_gradient(self):
        """Check objective derivative along a random direction.

        Return a dictionary containing the scaled error between the directional
        derivative of the objective in a random direction and the
        finite-difference approximation.
        """
        n = self.model.n
        fx = self.model.obj(self.x)
        gx = self.model.grad(self.x)
        errs = {}

        dx = np.random.standard_normal(n)
        dx /= norm(dx)
        xph = self.x.copy()
        xph += self.step * dx
        dfdx = (self.model.obj(xph) - fx) / self.step  # estimate
        gtdx = np.dot(gx, dx)                        # expected
        err = max(abs(dfdx - gtdx)/(1 + abs(gtdx)),
                  abs(dfdx - gtdx)/(1 + abs(dfdx)))

        self.log.debug('Objective directional derivative')
        self.log.debug(self.head3)
        line = self.d3fmt % ('', gtdx, dfdx, err)
        if err > self.tol:
            self.log.warn(line)
            errs['dir'] = dx
            errs['err'] = err
        else:
            self.log.debug(line)

        return errs

    def check_obj_gradient(self):
        """Check objective gradient using centered finite differences.

        Return a dictionary of gradient components for which the scaled error
        with the finite-difference approximation exceeds ``self.tol``.
        """
        model = self.model
        n = model.n
        model.obj(self.x)
        gx = model.grad(self.x)
        errs = {}

        self.log.debug('Objective gradient')
        self.log.debug(self.head)

        # Check partial derivatives in turn.
        xph = self.x.copy()
        xmh = self.x.copy()
        for i in xrange(n):
            xph[i] += self.step
            xmh[i] -= self.step
            dfdxi = (model.obj(xph) - model.obj(xmh)) / (2 * self.step)
            err = abs(gx[i] - dfdxi)/max(1, abs(dfdxi))
            xph[i] = xmh[i] = self.x[i]

            line = self.d1fmt % (0, i, gx[i], dfdxi, err)

            if err > self.tol:
                self.log.warn(line)
                errs[i] = err
            else:
                self.log.debug(line)

        return errs

    def check_obj_hessian(self):
        """Check objective Hessian using centered finite differences.

        Return a dictionary of Hessian components for which the scaled error
        with the finite-difference approximation exceeds ``self.tol``.
        """
        model = self.model
        n = model.n
        Hx = model.hess(self.x)
        errs = {}

        if isinstance(Hx, tuple):
            self.log.warn("Hessian format not supported, using operator")
            Hx = model.hop(self.x)

        if not hasattr(Hx, "__getitem__"):
            # Extract Hessian values via dot products.
            ei = np.zeros(n)
            ej = np.zeros(n)

        self.log.debug('Objective Hessian')

        # Check second partial derivatives in turn.
        xph = self.x.copy()
        xmh = self.x.copy()
        for i in xrange(n):
            xph[i] += self.step
            xmh[i] -= self.step
            dgdx = (model.grad(xph) - model.grad(xmh)) / (2 * self.step)
            xph[i] = xmh[i] = self.x[i]

            if not hasattr(Hx, "__getitem__"):
                ei[i] = 1

            for j in xrange(i + 1):
                dgjdxi = dgdx[j]

                if hasattr(Hx, "__getitem__"):
                    Hxij = Hx[i, j]
                else:
                    ej[j] = 1
                    Hxij = np.dot(ei, Hx * ej)
                    ej[j] = 0

                err = abs(Hxij - dgjdxi)/max(1, abs(dgjdxi))

                line = self.d2fmt % (0, i, j, Hxij, dgjdxi, err)
                if err > self.tol:
                    self.log.warn(line)
                    errs[(i, j)] = err
                else:
                    self.log.debug(line)

            if not hasattr(Hx, "__getitem__"):
                # Restore ei before the next iteration.
                ei[i] = 0

        return errs

    def check_con_jacobian(self):
        """Check constraints Jacobian using centered finite differences."""
        model = self.model
        n = model.n
        m = model.m
        if m == 0:
            return {}   # Problem is unconstrained.

        Jx = model.jac(self.x)
        errs = {}

        if isinstance(Jx, tuple):
            self.log.error("Jacobian format not supported, using operator")
            Jx = model.jop(self.x)

        if not hasattr(Jx, "__getitem__"):
            # Extract Jacobian values via dot products.
            ei = np.zeros(n)
            ej = np.zeros(m)

        self.log.debug('Constraints Jacobian')

        # Check partial derivatives of each constraint in turn.
        xph = self.x.copy()
        xmh = self.x.copy()
        for i in xrange(n):  # i = variable.
            xph[i] += self.step
            xmh[i] -= self.step
            dcdxi = (model.cons(xph) - model.cons(xmh)) / (2 * self.step)
            xph[i] = xmh[i] = self.x[i]

            if not hasattr(Jx, "__getitem__"):
                ei[i] = 1

            for j in xrange(m):  # j = constraint.
                dcjdxi = dcdxi[j]

                if hasattr(Jx, "__getitem__"):
                    Jxji = Jx[j, i]
                else:
                    ej[j] = 1
                    Jxji = np.dot(ej, Jx * ei)
                    ej[j] = 0

                err = abs(Jxji - dcjdxi) / max(1, abs(dcjdxi))

                line = self.d1fmt % (j + 1, i, Jxji, dcjdxi, err)
                if err > self.tol:
                    self.log.warn(line)
                    errs[(j, i)] = err
                else:
                    self.log.debug(line)

            if not hasattr(Jx, "__getitem__"):
                # Restore ei before the next iteration.
                ei[i] = 0

        return errs

    def check_con_hessians(self):
        """Check constraints Hessians using centered finite differences.

        Return a dictionary of dictionaries, the k-th of which contains the
        k-th Hessian components for which the scaled error with the
        finite-difference approximation exceeds ``self.tol``.
        """
        n = self.model.n
        m = self.model.m
        errs = {}

        self.log.debug('Constraints Hessians')

        # Check each Hessian in turn.
        y = np.zeros(m)
        xph = self.x.copy()
        xmh = self.x.copy()
        for k in xrange(m):
            y[k] = -1
            Hk = self.model.hess(self.x, y, obj_weight=0)
            y[k] = 0
            errs[k] = {}

            if isinstance(Hk, tuple):
                self.log.error("Hessian format not supported, using operator")
                Hk = self.model.hop(self.x, y, obj_weight=0)

            if not hasattr(Hk, "__getitem__"):
                # Extract Hessian values via dot products.
                ei = np.zeros(n)
                ej = np.zeros(n)

            # Check second partial derivatives in turn.
            for i in xrange(n):
                xph[i] += self.step
                xmh[i] -= self.step
                dgdx = (self.model.igrad(k, xph) -
                        self.model.igrad(k, xmh)) / (2 * self.step)
                xph[i] = xmh[i] = self.x[i]
                for j in xrange(i + 1):
                    dgjdxi = dgdx[j]

                    if hasattr(Hk, "__getitem__"):
                        Hkij = Hk[i, j]
                    else:
                        ej[j] = 1
                        Hkij = np.dot(ei, Hk * ej)
                        ej[j] = 0

                    err = abs(Hkij - dgjdxi) / max(1, abs(dgjdxi))

                    line = self.d2fmt % (k + 1, i, j, Hkij, dgjdxi, err)
                    if err > self.tol:
                        self.log.warn(line)
                        errs[k][(i, j)] = err
                    else:
                        self.log.debug(line)

                if not hasattr(Hk, "__getitem__"):
                    # Restore ei before the next iteration.
                    ei[i] = 0

        return errs
