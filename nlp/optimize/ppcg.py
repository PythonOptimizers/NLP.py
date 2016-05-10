# -*- coding: utf-8 -*-
"""An implementation of the projected conjugate gradient algorithm.

Described in

  N.I.M. Gould, M.E. Hribar and J. Nocedal,
  *On the Solution of Equality Constrained Quadratic Programming
  Problems Arising in Optimization*,
  SIAM Journal on Scientific Computing **23** (4), pp. 1376-1395, 2001.

with the addition of an optional trust-region constraint.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""
from math import sqrt
import numpy as np

from nlp.optimize.projKrylov import ProjectedKrylov
from nlp.tools.utils import to_boundary
from nlp.tools.timing import cputime

__docformat__ = 'restructuredtext'


class ProjectedCG(ProjectedKrylov):
    """A projected preconditioned conjugate-gradient algorithm."""

    def __init__(self, qp, **kwargs):
        u"""Solve the equality-constrained quadratic programming problem.

            minimize     cᵀx + ½ xᵀHx                           (1)
            subject to   Ax = b

        using the projected preconditioned conjugate-gradient algorithm.
        Possibly, there may be no linear equality constraints in the problem.

        The only mandatory input argument is `qp`, an instance of a
        :class:`QPModel`.

        This module may also be used to solve the equality-constrained
        trust-region subproblem

            minimize     cᵀx + ½ xᵀHx
            subject to   Ax = 0                                   (2)
                         sqrt(xᵀMx) ≤ Δ,

        where M is a symmetric positive definite scaling matrix and Δ > 0
        is a given trust-region radius. Note that b = 0 in the latter problem.
        For the time being, only M = I is implemented, i.e., the Euclidian norm
        is used. Specifying M is equivalent to specifying a preconditioner. See
        the keyword 'precon'.

        Equivalently, this module is appropriate for solving saddle-point
        systems of the form

            [ H   Aᵀ ] [ x ] = [ -c ]                                      (3)
            [ A   0  ] [ y ]   [  b ]

        where H is a symmetric matrix. If H is positive definite on the
        nullspace of A, then (1) and (3) are equivalent. Otherwise, it is
        possible that no finite solution exists or that there are an infinite
        number of them. A 'trust-region radius' must then be
        specified. If any of the latter two cases apply, this module computes
        an initial x0 satisfying  A x0 = b, solves problem (2) with appropriate
        values of c and H and returns the final solution x+x0.

        Unless A is explicitly specified in `qp`, we assume that there are no
        equality constraints. In this case, the method reduces to the
        well-known conjugate gradient method. The right-hand side `b` is taken
        as `qp.Lcon`.

        The symmetric matrix H need not be given explicitly but may be a linear
        operator.

        A preconditioner may be given by using the precon keyword. For the time
        being, the preconditioner should be given in assembled form and may not
        be an operator. The preconditioner G is used to assemble and factorize
        the symmetric indefinite projection matrix

            [ G   Aᵀ ]
            [ A   0  ].

        The user should keep in mind that G must be relatively sparse and must
        be positive definite over the nullspace of A. If no preconditioner is
        given, everything happens as if G = I (the identity matrix) were given.

        The algorithm stops as soon as the norm of the projected gradient
        falls under

            max(abstol, reltol * g0)

        where

            abstol      is an absolute stopping tolerance
            reltol      is a relative stopping tolerance
            g0          is the norm of the initial projected gradient
                        (or of the preconditioned projected gradient, if a
                         preconditioner is given.)

        :keywords:
            :precon: a preconditioner G (given explicitly) (Identity)
            :proj: an existing factorization of the projection matrix
                   conforming to the LBLContext (None)
            :abstol: the absolute stopping tolerance (1.0e-8)
            :reltol: the relative stopping tolerance (1.0e-6)
            :maxiter: the maximum number of iterations (2n)
            :max_itref: the maximum number of iterative refinement steps (3)
            :itref_tol: the required threshold for the residual after
                        iterative refinement (1.0e-6)
            :radius: trust-region radius (None)
            :btol: fraction-to-the-boundary factor (None)
            :cur_iter: a vector related to btol (see below) (None)
            :factorize: set to `False` if calling again with the same
                        constraint matrix `A` (True)
            :debug: a boolean indicating debug/verbose mode (False)

        If specified, a positive factor `btol` will cause the algorithm to
        enforce all conjugate gradient iterates to satisfy  sk >= btol.
        Typically, in an interior-point methods calling `ppcg()`, the step
        computed at iteration k must be such that
                dk ≥ - τ xₖ
        where 0 < τ < 1, so that
                xₖ + dₖ ≥ (1-τ) xₖ,
        which prevents the iterates from getting too close to the boundary of
        the nonnegative orthant. In this type of setting, btol should be set
        to
                btol = τ
        and the vector cur_iter should be set to
                cur_iter = xₖ.
        This ensures that no copy of xₖ occurs and only a pointer to xₖ is
        used.

        Upon completion, a few members of the instance are set so a status
        check can be performed. The most important situations are:

        * A point was found where the residual is sufficiently small (whether
          no trust region was present, or its boundary was not encountered).
          This can only happen when `H` is second-order sufficient.
          In this case `on_boundary` is set to `False` and `inf_descent` is set
          to `False`.
        * No trust region is present but the problem is not second-order
          sufficient. In this case, an infinite descent direction has been
          identified: `inf_descent` is set to `True` and `dir` contains the
          infinite descent direction. `on_boundary` is set to `False`.
        * A trust-region is present and its boundary was hit. If no infinite
          descent direction has been discovered, `inf_descent` is set to
          `False`. Otherwise, it is set to `True`. In both cases, `on_boundary`
          is set to `True`.

        Reference
        ---------

        .. [GHN01]  N.I.M. Gould, M.E. Hribar and J. Nocedal, *On the Solution
                    of Equality Constrained Quadratic Programming Problems
                    Arising in Optimization*, SIAM Journal on Scientific
                    Computing **23**(4), pp. 1376-1395, 2001.
        """
        super(ProjectedCG, self).__init__(qp.c, qp.H, A=qp.A, **kwargs)

        self.qp = qp
        self.prefix = 'Ppcg: '
        self.name = 'Projected CG'
        self.radius = kwargs.get('radius', None)

        if self.radius is not None and self.radius <= 0.0:
            raise ValueError('Radius must be a positive real number')

        self.btol = kwargs.get('btol', None)
        self.cur_iter = kwargs.get('cur_iter', None)
        self.precon = kwargs.get('precon', None)

        # Initializations
        self.x_feasible = None
        self.x = np.zeros(self.n, 'd')
        self.step = self.x  # Shortcut for consistency with TruncatedCG
        self.v = None
        self.resid_norm = None
        self.resid_norm0 = None
        self.rhs = np.zeros(self.n + self.m, 'd')
        self.iter = self.n_matvec = 0
        self.inf_descent_dir = None
        self.inf_descent = False  # Direction of infinity descent
        self.x_norm2 = 0.0        # Squared step norm, not counting x_feasible
        self.step_norm = 0.0      # Shortcut for consistency with TruncatedCG
        self.on_boundary = False
        self.status = None

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s'
        self.header = self.hd_fmt % ('Iter', '<r,g>', 'curv')
        self.fmt1 = ' %-5d  %9.2e'
        self.fmt = ' %-5d  %9.2e  %8.2e'

    def ftb(self, s, p):
        u"""Fraction to the boundary.

        If fraction-to-the-boundary rule is to be enforced, compute step
        length to satisfy  s + t*p ≥ btol * cur_iter.
        """
        neg_idx = np.where(p < 0.0)[0]
        step_len = 1.0
        for i in neg_idx:
            step_len = min(step_len,
                           -(self.btol * self.cur_iter[i] + s[i]) / p[i])
        return step_len

    def solve(self):
        """Solve."""
        n = self.n
        x_norm2 = 0.0  # Squared norm of current iterate x, not counting x_feas

        # Obtain initial projected residual
        self.t_solve = cputime()
        if self.qp.A is not None:
            if self.factorize and not self.factorized:
                self.perform_factorization()
            if self.b is not None:
                self.rhs[:n] = self.qp.grad(self.x_feasible)
                self.rhs[n:] = 0.0
            else:
                self.rhs[:n] = self.qp.c
            self.proj.solve(self.rhs)
            r = g = self.proj.x[:n]
            self.v = self.proj.x[n:]

            # self.CheckAccurate()

        else:
            g = self.qp.c
            r = g.copy()

        # Initialize search direction
        p = -g
        pHp = None

        self.resid_norm0 = np.dot(r, g)
        rg = self.resid_norm0
        threshold = max(self.abstol, self.reltol * sqrt(self.resid_norm0))
        iter = 0
        on_boundary = False

        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        self.log.info(self.fmt1, iter, rg)

        while sqrt(rg) > threshold and iter < self.maxiter and not on_boundary:

            Hp = self.qp.H * p
            pHp = np.dot(p, Hp)

            # Display current iteration info
            self.log.info(self.fmt, iter, rg, pHp)

            if self.radius is not None:
                # Compute steplength to the boundary
                sigma = to_boundary(self.x, p, self.radius, xx=x_norm2)
                if (self.btol is not None) and (self.cur_iter is not None):
                    sigma = min(sigma, self.ftb(self.x, p))
            elif pHp <= 0.0:
                self.log.error('Problem is not second-order sufficient')
                status = 'problem not SOS'
                self.inf_descent = True
                self.inf_descent_dir = p
                continue

            alpha = rg / pHp if pHp != 0.0 else np.inf

            if self.radius is not None and (pHp <= 0.0 or alpha > sigma):
                # p is a direction of singularity or negative curvature or
                # next iterate will lie past the boundary of the trust region
                # Move to boundary of trust-region
                self.x += sigma * p
                x_norm2 = self.radius * self.radius
                status = u'on boundary (σ = %g)' % sigma
                self.inf_descent = (pHp <= 0.0)
                on_boundary = True
                continue

            # Make sure nonnegativity bounds remain enforced, if requested
            if (self.btol is not None) and (self.cur_iter is not None):
                step_bnd = self.ftb(self.x, p)
                if step_bnd < alpha:
                    self.x += step_bnd * p
                    status = 'on boundary'
                    on_boundary = True
                    continue

            # Move on
            self.x += alpha * p
            r += alpha * Hp

            if self.qp.A is not None:
                # Project current residual
                self.rhs[:n] = r
                self.proj.solve(self.rhs)

                # Perform actual iterative refinement, if necessary
                # self.proj.refine(self.rhs, nitref=self.max_itref,
                #                  tol=self.itref_tol)

                # Obtain new projected gradient
                g = self.proj.x[:n]
                if self.precon is not None:
                    # Prepare for iterative semi-refinement
                    self.qp.jtprod(self.proj.x[n:], self.v)
            else:
                g = r

            rg_next = np.dot(r, g)
            beta = rg_next / rg
            p = -g + beta * p
            if self.precon is not None:
                # Perform iterative semi-refinement
                r = r - self.v
            else:
                r = g
            rg = rg_next

            if self.radius is not None:
                x_norm2 = np.dot(self.x, self.x)
            iter += 1

        # Output info about the last iteration
        if iter > 0:
            self.log.info(self.fmt, iter, rg, pHp)

        # Obtain final solution x
        self.x_norm2 = x_norm2
        self.step_norm = sqrt(x_norm2)
        if self.x_feasible is not None:
            self.x += self.x_feasible

        if self.qp.A is not None:
            # Find (weighted) least-squares Lagrange multipliers
            self.rhs[:n] = - self.qp.grad(self.x)
            self.rhs[n:] = 0.0
            self.proj.solve(self.rhs)
            self.v = self.proj.x[n:].copy()

        self.t_solve = cputime() - self.t_solve

        self.step = self.x  # Alias for consistency with TruncatedCG.
        self.on_boundary = on_boundary
        self.converged = (iter < self.maxiter)
        if iter < self.maxiter and not on_boundary:
            status = 'residual small'
        elif iter >= self.maxiter:
            status = 'max iter'
        self.iter = iter
        self.n_matvec = iter
        self.resid_norm = sqrt(rg)
        self.status = status

        return
