# -*- coding: utf-8 -*-
"""Class definition for Trust-Region Algorithm and Management."""

import numpy as np

__docformat__ = 'restructuredtext'


class TrustRegion(object):
    """A trust-region management class."""

    def __init__(self, **kwargs):
        """Instantiate an object allowing management of a trust region.

        :keywords:

            :radius:        Initial trust-region radius (default: 1.0)
            :eta1:          Step acceptance threshold   (default: 0.01)
            :eta2:          Radius increase threshold   (default: 0.99)
            :gamma1:        Radius decrease factor      (default: 1/3)
            :gamma2:        Radius increase factor      (default: 2.5)

        Subclass and override :meth:`update_radius` to implement custom
        trust-region management rules.

        See, e.g.,

        A. R. Conn, N. I. M. Gould and Ph. L. Toint, Trust-Region Methods,
        MP01 MPS-SIAM Series on Optimization, 2000.
        """
        self.radius = self.radius0 = kwargs.get('radius', 1.0)
        self.radius_max = 1.0e+10
        self.eta1 = kwargs.get('eta1', 0.01)    # Step acceptance threshold
        self.eta2 = kwargs.get('eta2', 0.99)    # Radius increase threshold
        self.gamma1 = kwargs.get('gamma1', 1.0 / 3)  # Radius decrease factor
        self.gamma2 = kwargs.get('gamma2', 2.5)    # Radius increase factor
        self.eps = np.finfo(np.double).eps  # Machine epsilon.

    def ratio(self, f, f_trial, m, check_positive=True):
        """Compute the ratio of actual versus predicted reduction.

        rho = (f - f_trial)/(-m).
        """
        pred = -m + max(1.0, abs(f)) * 10.0 * self.eps
        ared = f - f_trial + max(1.0, abs(f)) * 10.0 * self.eps
        if pred > 0 or not check_positive:
            return ared / pred
        else:
            # Error: Negative predicted reduction
            msg = 'TrustRegion:: Nonpositive predicted reduction: %8.1e' % pred
            raise ValueError(msg)
            return None

    def update_radius(self, ratio, step_norm):
        """Update the trust-region radius.

        The rule implemented by this method is:

        radius = gamma1 * step_norm      if ared/pred <  eta1
        radius = gamma2 * radius         if ared/pred >= eta2
        radius unchanged otherwise,

        where ared/pred is the quotient computed by :meth:`ratio`.
        """
        if ratio < self.eta1:
            self.radius = self.gamma1 * step_norm
        elif ratio >= self.eta2:
            self.radius = min(max(self.radius, self.gamma2 * step_norm),
                              self.radius_max)

    def reset_radius(self):
        """Reset radius to original value."""
        self.radius = self.radius0


class GeneralizedTrustRegion(TrustRegion):
    """A more general trust-region management class.

    Subclass of `TrustRegion` implementing a more flexible update of the
    trust-region radius.

    See :meth:`update_radius` for more details.
    """

    def __init__(self, **kwargs):
        """Instantiate an object allowing management of a trust region.

        :keywords:
            :radius:        Initial trust-region radius (default: 1.0)
        """
        self.radius = self.radius0 = kwargs.get('radius', 1.0)
        self.radius_max = 1.0e+10
        self.eta0 = 1e-4
        self.eta1 = 0.25
        self.eta2 = 0.75
        self.gamma1 = 0.25
        self.gamma2 = 0.5
        self.gamma3 = 4.0
        self.eps = np.finfo(np.double).eps  # Machine epsilon.

    def update_radius(self, ratio, step_norm, alpha):
        u"""Update the trust-region radius.

        The rule implemented by this method is:

        radius = min(max(α, γ1) * step_norm, γ2 radius       if ratio <= η0
        radius = max(γ1*radius, min(α*step_norm, γ2*radius)) if ratio ∈ (η0,η1]
        radius = max(γ1*radius,
                     min(α*step_norm, γ3*self.radius))       if ratio ∈ (η1,η2]
        radius = max(radius, min(α*step_norm, γ3*radius))    if ratio > η2.
        """
        if ratio <= self.eta0:
            self.radius = min(max(alpha, self.gamma1) * step_norm,
                              self.gamma2 * self.radius)
        elif ratio <= self.eta1:
            self.radius = max(self.gamma1 * self.radius,
                              min(alpha * step_norm,
                                  self.gamma2 * self.radius))
        elif ratio <= self.eta2:
            self.radius = max(self.gamma1 * self.radius,
                              min(alpha * step_norm,
                                  self.gamma3 * self.radius))
        else:
            self.radius = max(self.radius, min(alpha * step_norm,
                                               self.gamma3 * self.radius))


class TrustRegionSolver(object):
    """A generic class for implementing solvers for the trust-region subproblem.

    minimize q(d)  subject to  ||d|| <= radius,

    where q(d) is a quadratic function, not necessarily convex.

    The trust-region constraint `||d|| <= radius` can be defined in any
    norm although most derived classes currently implement the Euclidian
    norm only. Note however that any elliptical norm may be used via a
    preconditioner.

    For more information on trust-region methods, see

    A. R. Conn, N. I. M. Gould and Ph. L. Toint, Trust-Region Methods,
    MP01 MPS-SIAM Series on Optimization, 2000.
    """

    def __init__(self, qp, cg_solver, **kwargs):
        """Basic solver for a quadratic trust-region subproblem.

        :keywords:

          :qp: a ``QPModel`` instance
          :cg_solver: a `class` to be instantiated and used as solver for the
                      trust-region problem.
        """
        self._qp = qp
        self._cg_solver = cg_solver(qp, **kwargs)
        self._niter = 0
        self._step_norm = 0.0
        self._step = None
        self._m = None  # Model value at candidate solution
        self.status = ""

    @property
    def qp(self):
        """Return the ``QPModel`` instance."""
        return self._qp

    @property
    def niter(self):
        """Return the number of iterations made by the solver."""
        return self._niter

    @property
    def step_norm(self):
        """Return the norm of the step computed by the solver."""
        return self._step_norm

    @property
    def step(self):
        """Return the step computed by the solver."""
        return self._step

    @property
    def m(self):
        """Return the value of the quadratic model."""
        return self._m

    @property
    def model_value(self):
        """Return the value of the quadratic model."""
        return self._m

    def solve(self, *args, **kwargs):
        """Solve the trust-region subproblem."""
        self._cg_solver.solve(*args, **kwargs)
        self._niter = self._cg_solver.niter
        self._step_norm = self._cg_solver.step_norm
        self._step = self._cg_solver.step
        self._m = self.qp.obj(self.step)        # Compute model reduction.
        self.status = self._cg_solver.status
