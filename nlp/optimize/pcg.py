"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM Journal on Numerical Analysis **20** (3), pp. 626-637,
  1983.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

from nlp.tools.utils import to_boundary
from nlp.tools.exceptions import UserExitRequest
import numpy as np
from math import sqrt
import logging

__docformat__ = 'restructuredtext'


class TruncatedCG(object):

    def __init__(self, qp, **kwargs):
        """
        Solve the quadratic trust-region subproblem

          minimize    g's + 1/2 s'Hs
          subject to  s's  <=  radius

        by means of the truncated conjugate gradient algorithm (aka the
        Steihaug-Toint algorithm). The notation `x'y` denotes the dot
        product of vectors `x` and `y`.

        :parameters:
            :qp:           an instance of the :class:`QPModel` class.
                           The Hessian H must be a symmetric linear
                           operator of appropriate size, but not necessarily
                           positive definite.
            :logger_name:  name of a logger object that can be used during the
                           iterations                         (default None)

        :returns:

          Upon return, the following attributes are set:

          :step:       final step,
          :niter:      number of iterations,
          :step_norm:  Euclidian norm of the step,
          :dir:        direction of infinite descent (if radius=None and
                       H is not positive definite),
          :onBoundary: set to True if trust-region boundary was hit,
          :infDescent: set to True if a direction of infinite descent was found

        The algorithm stops as soon as the preconditioned norm of the gradient
        falls under

            max( abstol, reltol * g0 )

        where g0 is the preconditioned norm of the initial gradient (or the
        Euclidian norm if no preconditioner is given), or as soon as the
        iterates cross the boundary of the trust region.
        """

        self.qp = qp
        self.n = qp.c.shape[0]

        self.prefix = 'Pcg: '
        self.name = 'Truncated CG'

        self.status = '?'
        self.onBoundary = False
        self.step = None
        self.step_norm = 0.0
        self.niter = 0
        self.dir = None
        self.qval = None
        self.pHp = None

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlp.trcg')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt0 = ' %-5d  %9.2e'
        self.fmt = self.fmt0 + '  %8.2e'

        return

    def post_iteration(self, *args, **kwargs):
        """
        Subclass and override this method to implement custom post-iteration
        actions. This method will be called at the end of each CG iteration.
        """
        pass

    def solve(self, **kwargs):
        """
        Solve the trust-region subproblem.

        :keywords:

          :s0:         initial guess (default: [0,0,...,0]),
          :radius:     the trust-region radius (default: None),
          :abstol:     absolute stopping tolerance (default: 1.0e-8),
          :reltol:     relative stopping tolerance (default: 1.0e-6),
          :maxiter:    maximum number of iterations (default: 2n),
          :prec:       a user-defined preconditioner.
        """

        radius = kwargs.get('radius', None)
        abstol = kwargs.get('absol', 1.0e-8)
        reltol = kwargs.get('reltol', 1.0e-6)
        maxiter = kwargs.get('maxiter', 2 * self.n)
        prec = kwargs.get('prec', lambda v: v)

        qp = self.qp
        n = qp.n
        H = qp.H

        # Initialization
        if 's0' in kwargs:
            s = kwargs['s0']
            snorm2 = np.linalg.norm(s)
        else:
            s = np.zeros(n)
            snorm2 = 0.0

        self.qval = qp.obj(s)
        r = qp.grad(s)

        y = prec(r)
        ry = np.dot(r, y)
        sqrtry = sqrt(ry)

        stop_tol = max(abstol, reltol * sqrtry)
        k = 0

        exitOptimal = sqrtry <= stop_tol
        exitIter = k > maxiter
        exitUser = False

        p = -y

        onBoundary = False
        infDescent = False

        self.log.info(self.header)
        self.log.info('-' * len(self.header))

        while not (exitOptimal or exitIter or exitUser) and \
                not onBoundary and not infDescent:

            k += 1
            Hp = H * p
            pHp = np.dot(p, Hp)

            self.log.info(self.fmt % (k, ry, pHp))

            # Compute steplength to the boundary.
            if radius is not None:
                sigma = to_boundary(s, p, radius, xx=snorm2)

            if pHp <= 0 and radius is None:
                # p is direction of singularity or negative curvature.
                self.status = 'infinite descent'
                snorm2 = 0
                self.dir = p
                self.pHp = pHp
                infDescent = True
                continue

            # Compute CG steplength.
            alpha = ry / pHp if pHp != 0 else np.inf

            if radius is not None and (pHp <= 0 or alpha > sigma):
                # p leads past the trust-region boundary. Move to the boundary.
                s += sigma * p
                snorm2 = radius * radius
                self.status = 'trust-region boundary active'
                onBoundary = True
                continue

            self.qval += alpha * np.dot(r, p) + 0.5 * alpha**2 * pHp
            self.ds = alpha * p
            self.dr = alpha * Hp

            # Move to next iterate.
            s += self.ds
            r += self.dr
            y = prec(r)
            ry_next = np.dot(r, y)
            beta = ry_next / ry
            p *= beta
            p -= y  # p = -y + beta * p
            ry = ry_next

            # Transfer useful quantities for post iteration.
            self.pHp = pHp
            self.r = r
            self.y = y
            self.p = p
            self.step = s
            self.step_norm2 = snorm2
            self.ry = ry
            self.alpha = alpha
            self.beta = beta

            sqrtry = sqrt(ry)
            snorm2 = np.dot(s, s)

            try:
                self.post_iteration()
            except UserExitRequest:
                self.status = 'usr'
                exitUser = True

            exitIter = k >= maxiter
            exitOptimal = sqrtry <= stop_tol

        # Output info about the last iteration.
        if k > 0:
            self.log.info(self.fmt % (k, ry, pHp))
        else:
            self.log.info(self.fmt0 % (k, ry))

        if k >= maxiter:
            self.status = 'max iter'
        elif not onBoundary and not infDescent and not exitUser:
            self.status = 'residual small'
        self.log.info(self.status)
        self.step = s
        self.niter = k
        self.step_norm = sqrt(snorm2)
        self.onBoundary = onBoundary
        self.infDescent = infDescent
        return
