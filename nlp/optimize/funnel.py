"""An implementation of the trust-funnel method."""
from math import sqrt
import numpy as np
import logging

from nlp.model.nlpmodel import NLPModel, QPModel
from nlp.ls.linesearch import ArmijoLineSearch
from nlp.optimize.ppcg import ProjectedCG, LSTRFramework
from nlp.optimize.lsqr import LSQRFramework
from nlp.tools.timing import cputime


__docformat__ = 'restructuredtext'


class Funnel(object):
    """A trust-funnel framework for equality-constrained optimization.

    D. Orban and N. I. M. Gould, from N. I. M. Gould's original Matlab
    implementation.
    """

    def __init__(self, model, **kwargs):
        """Instantiate a trust-funnel framework.

        :parameters:
            :model: `NLPModel` instance (should be equality-constrained only).

        :keywords:
            :atol: Absolute stopping tolerance
            :delta_f: Initial trust-region radius for the objective model
            :delta_c: Initial trust-region radius for the constraints model
            :iterative_solver: solver used to find steps (0=gltr, 1=gmres)
            :maxit: maximum number of iterations allowed
            :stop_p: required accuracy for ||c||
            :stop_d: required accuracy for ||g+J'y||
            :maxit_refine: maximum number of iterative refinements per solve
        """
        # Bail out if model is not a NLPModel instance.
        if not isinstance(model, NLPModel):
            msg = 'Input problem must be a NLPModel instance.'
            raise ValueError(msg)

        # Bail out if problem has bounds or general inequality constraints.
        if model.nlowerB + model.nupperB + model.nrangeB:
            msg = 'Problem has bound constraints.'
            raise ValueError(msg)

        if model.nlowerC + model.nupperC + model.nrangeC:
            msg = 'Problem has general inequality constraints.'
            raise ValueError(msg)

        self.model = model
        self.x = model.x0.copy()
        self.f0 = self.f = model.obj(self.x)
        self.p_resid = self.d_resid = None

        # Members to memorize old and current Jacobian matrices if needed.
        self.save_J = False
        self.J = self.J_old = None

        self.optimal = False

        self.niter = 0
        self.tsolve = 0.0

        # Set default options.
        self.atol = kwargs.get('atol', 1.0e-8)
        self.delta_f_0 = self.delta_f = kwargs.get('delta_f', 1.0)
        self.delta_c_0 = self.delta_c = kwargs.get('delta_c', 1.0)
        self.iterative_solver = kwargs.get('iterative_solver', 0)
        self.maxit = kwargs.get('maxit', 200)
        self.stop_p = kwargs.get('stop_p', model.stop_p)
        self.stop_d = kwargs.get('stop_d', model.stop_d)
        self.maxit_refine = kwargs.get('maxit_refine', 0)

        self.hdr_fmt = '%4s   %2s %9s %8s %8s %8s %8s %6s %6s'
        self.hdr = self.hdr_fmt % ('iter', 'NT', 'f', '|c|', '|g+J''y|',
                                   'delta_f', 'delta_c', 'thetaMax', 'CG')
        self.linefmt1 = '%4d%c%c %c%c %9.2e %8.2e %8.2e ' + \
                        '%8.2e %8.2e %8.2e %6.0f'
        self.linefmt = '%4d%c%c %c%c %9.2e %8.2e %8.2e ' + \
                       '%8.2e %8.2e %8.2e %6.0f'

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'funnel.solver')
        self.log = logging.getLogger(logger_name)

        # Display initial information.
        self.display_basic_info()
        return

    def display_basic_info(self):
        """Display basic info about current problem."""
        model = self.model
        self.log.info('Problem %s' % model.name)
        self.log.info('Linear  equality constraints: %d' % model.nlin)
        self.log.info('General equality constraints: %d' %
                      (model.nequalC - model.nlin))
        self.log.info('Primal stopping tolerance: %6.1e' % self.stop_p)
        self.log.info('Dual   stopping tolerance: %6.1e' % self.stop_d)
        self.log.info('Key:')
        self.log.info('iter type   : [f] or [c]')
        self.log.info('overall step: [s]uccessful, [v]ery successful, ' +
                      '[u]nsuccessful or (2)nd-order')
        self.log.info('normal  step: [0] (none), [r]esidual small, ' +
                      '[b]oundary or [?] (other)')
        self.log.info('tangent step: [0] (none), [r]esidual small, ' +
                      '[b]oundary, [-] neg. curvature, ' +
                      '[>] max iter or [?] (other)')
        return

    def forcing(self, k, val):
        """Return forcing term number `k`."""
        if k == 1:
            return 1.0e-2 * min(1, min(abs(val), val * val))
        if k == 2:
            return 1.0e-2 * min(1, min(abs(val), val * val))
        return 1.0e-2 * min(1, min(abs(val), val * val))

    def post_iteration(self, **kwargs):
        """Perform work at the end of an iteration.

        Override this method. For example, use this method for updating a LBFGS
        Hessian.
        """
        pass

    def lsq(self, A, b, reg=0.0, radius=None, **kwargs):
        """Solve the linear least-squares problem in the variable x.

            minimize |Ax - b| + reg * |x|

        in the Euclidian norm with LSQR. Optionally, x may be subject to a
        Euclidian-norm trust-region constraint |x| <= radius.
        This function returns (x, x_norm, status).
        """
        LSQR = LSQRFramework(A)
        LSQR.solve(b, radius=radius, damp=reg, show=False)
        return (LSQR.x, LSQR.xnorm, LSQR.status)

    def nyf(self, x, f, f_trial, g, step, bkmax=5, armijo=1.0 - 4):
        """Perform a simple backtracking linesearch on the objective function.

        Linesearch is performed starting from `x` along direction `step`.
        Here, `f` and `f_trial` are the objective value at `x` and `x + step`,
        respectively, and `g` is the gradient of the objective at `x`.

        Return (x, f, steplength), where `x + steplength * step` satisfies
        the Armijo condition and `f` is the objective value at this new point.
        """
        slope = np.dot(g, step)
        return self.als_f.search(self.model.obj, x, step, slope,
                                 bkmax=bkmax, f=f, f_trial=f_trial)

    def nyc(self, x, theta, theta_trial, c, grad_theta, step, bkmax=5,
            armijo=1.0e-4):
        u"""Perform a backtracking linesearch on the infeasibility measure.

        Linesearch is performed on the infeasibility measure defined by
            Θ(x) = 1/2 |c(x)|^2
        starting from `x` along direction `step`.

        Here, `theta` and `theta_trial` are the infeasibility at `x` and
        `x + step`, respectively, `c` is the vector of constraints at `x`, and
        `grad_theta` is the gradient of the infeasibility measure at `x`.

        Note that  ∇ Θ(x) = J(x)^T  c(x).

        Return (x, c, theta, steplength), where `x + steplength + step`
        satisifes the Armijo condition, and `c` and `theta` are the vector of
        constraints and the infeasibility at this new point, respectively.
        """
        slope = np.dot(grad_theta, step)

        def function_theta(x):
            c = self.model.cons(x)
            theta = 0.5 * np.dot(c, c)
            return theta

        (x_trial, theta_trial, alpha) = self.asl_c.search(function_theta, x,
                                                          step, slope,
                                                          bkmax=bkmax, f=theta,
                                                          f_trial=theta_trial)

        c_trial = self.model.cons(x_trial)
        return (x_trial, c_trial, theta_trial, alpha)

    def solve(self, **kwargs):
        """Solve current problem with trust-funnel framework.

        :keywords:
            :ny: Enable Nocedal-Yuan backtracking linesearch.

        :returns:
            This method sets the following members of the instance:
            :f: Final objective value
            :optimal: Flag indicating whether normal stopping conditions were
                      attained
            :p_resid: Final primal residual
            :d_resid: Final dual residual
            :niter: Total number of iterations
            :tsolve: Solve time.
        """
        ny = kwargs.get('ny', True)
        reg = kwargs.get('reg', 0.0)
        # ny = False

        tsolve = cputime()

        # Set some shortcuts.
        model = self.model
        n = model.n
        m = model.m
        x = self.x
        f = self.f
        c = model.cons_pos(x)
        y = model.pi0.copy()
        self.it = 0

        # Initialize some constants.
        kappa_n = 1.0e+2   # Factor of p_norm in normal step TR radius.
        kappa_b = 0.99     # Fraction of TR to compute tangential step.
        kappa_delta = 0.1  # Progress factor to compute tangential step.

        # Trust-region parameters.
        eta_1 = 1.0e-5
        eta_2 = 0.95
        eta_3 = 0.5
        gamma_1 = 0.25
        gamma_3 = 2.5

        kappa_tx1 = 0.9  # Factor of theta_max in max acceptable infeasibility.
        kappa_tx2 = 0.5  # Convex combination factor of theta and theta_trial.

        # Linesearches
        self.als_f = ArmijoLineSearch(tfactor=5. / 6)  # objective function
        self.als_c = ArmijoLineSearch(tfactor=5. / 6)  # infeasibility measure

        # Compute constraint violation.
        theta = 0.5 * np.dot(c, c)

        # Set initial funnel radius.
        kappa_ca = 1.0e+3    # Max initial funnel radius.
        kappa_cr = 2.0       # Infeasibility tolerance factor.
        theta_max = max(kappa_ca, kappa_cr * theta)

        # Evaluate first-order derivatives.
        g = model.grad(x)
        J = model.jac(x)
        Jop = model.jop(x)

        # Initial radius for f- and c-iterations.
        delta_f = max(self.delta_f, .1 * np.linalg.norm(g))
        delta_c = max(self.delta_c, .1 * sqrt(2 * theta))

        # Reset initial multipliers to least-squares estimates by
        # approximately solving:
        #   [ I   J' ] [ w ]   [ -g ]
        #   [ J   0  ] [ y ] = [  0 ].
        # This is equivalent to solving
        #   minimize |g + J'y|.
        if m > 0:
            y, _, _ = self.lsq(Jop.T, -g, reg=reg)

        p_norm = c_norm = 0
        if m > 0:
            p_norm = np.linalg.norm(c)
            c_norm = np.linalg.norm(c, np.inf)
            grad_lag = g + Jop.T * y
        else:
            grad_lag = g.copy()
        d_norm = np.linalg.norm(grad_lag) / (1 + np.linalg.norm(y))

        # Display current info if requested.
        self.log.info(self.hdr)
        self.log.info(self.linefmt1 % (0, ' ', ' ', ' ', ' ', f, p_norm,
                                       d_norm, delta_f, delta_c, theta_max, 0))

        # Compute primal stopping tolerance.
        stop_p = max(self.atol, self.stop_p * p_norm)
        self.log.debug('p_norm=%7.1e, c_norm=%7.1e, d_norm=%7.1e' %
                       (p_norm, c_norm, d_norm))

        optimal = (p_norm <= stop_p) and (d_norm <= self.stop_d)
        self.log.debug('optimal: %s' % repr(optimal))

        # Start of main iteration.
        while not optimal and (self.it < self.maxit):

            self.it += 1
            delta = min(delta_f, delta_c)
            cgiter = 0

            # 1. Compute normal step as an (approximate) solution to
            #    minimize |c + J n|  subject to  |n| <= min(delta_c, kN |c|).

            if self.it > 1 and \
                    p_norm <= stop_p and \
                    d_norm >= 1.0e+4 * self.stop_d:

                self.log.debug('Setting step_n=0 b/c must work on optimality')
                step_n = np.zeros(n)
                step_n_norm = 0.0
                status_n = '0'
                m_xpn = 0       # Model value at x+n.

            else:

                step_n_max = min(delta_c, kappa_n * p_norm)
                step_n, step_n_norm, lsq_status = self.lsq(Jop, -c,
                                                           radius=step_n_max,
                                                           reg=reg)

                if lsq_status == 'residual small':
                    status_n = 'r'
                elif lsq_status == 'trust-region boundary active':
                    status_n = 'b'
                else:
                    status_n = '?'

                # Evaluate the model of the obective after the normal step.
                _Hv = model.hprod(x, -y, step_n)  # H*step_n
                m_xpn = np.dot(g, step_n) + 0.5 * np.dot(step_n, _Hv)

            self.log.debug('Normal step norm = %8.2e' % step_n_norm)
            self.log.debug('Model value: %9.2e' % m_xpn)

            # 2. Compute tangential step if normal step is not too long.

            if step_n_norm <= kappa_b * delta:

                # 2.1. Compute Lagrange multiplier estimates and dual residuals
                #      by minimizing |(g + H n) + J'y|

                if step_n_norm == 0.0:
                    g_n = g   # Just a pointer ; g will not be modified below.
                else:
                    g_n = g + _Hv

                y_new, y_norm, _ = self.lsq(Jop.T, -g_n, reg=reg)
                r = g_n + Jop.T * y_new

                # Here Nick does iterative refinement to improve r and y_new...

                # Compute dual optimality measure.
                resid_norm = np.linalg.norm(r)
                pi = 0.0
                if resid_norm > 0:
                    pi = abs(np.dot(g_n, r)) / resid_norm

                # 2.2. If the dual residuals are large, compute a suitable
                #      tangential step as a solution to:
                #      minimize    g't + 1/2 t' H t
                #      subject to  Jt = 0, |n+t| <= delta.

                if pi > self.forcing(3, theta):

                    self.log.debug('Computing step_t...')
                    delta_within = delta - step_n_norm

                    Hop = model.hop(x, -y_new)

                    qp = QPModel(g_n, Hop, A=J.matrix if m > 0 else None)
                    # PPCG = ProjectedCG(g_n, Hop,
                    #                    A=J.matrix if m > 0 else None,
                    #                    radius=delta_within, dreg=reg)
                    PPCG = ProjectedCG(qp, radius=delta_within, dreg=reg)
                    PPCG.Solve()
                    step_t = PPCG.step
                    step_t_norm = PPCG.stepNorm
                    cgiter = PPCG.iter

                    self.log.debug('|t| = %8.2e' % step_t_norm)

                    if PPCG.status == 'residual small':
                        status_t = 'r'
                    elif PPCG.onBoundary and not PPCG.infDescent:
                        status_t = 'b'
                    elif PPCG.infDescent:
                        status_t = '-'
                    elif PPCG.status == 'max iter':
                        status_t = '>'
                    else:
                        status_t = '?'

                    # Compute total step and model decrease.
                    step = step_n + step_t
                    step_norm = np.linalg.norm(step)
                    # _Hv = model.hprod(x, -y, step)    # y or y_new?
                    # m_xps = np.dot(g, step) + 0.5 * np.dot(step, _Hv)
                    m_xps = qp.obj(step)

                else:

                    self.log.debug('Setting step_t=0 b/c pi sufficiently' +
                                   'small')
                    step_t_norm = 0
                    status_t = '0'
                    step = step_n
                    step_norm = step_n_norm
                    m_xps = m_xpn

                y = y_new

            else:

                # No need to compute a tangential step.
                self.log.debug('Setting step_t=0 b/c normal step too large')
                status_t = '0'
                y = np.zeros(m)
                step_t_norm = 0.0
                step = step_n
                step_norm = step_n_norm
                m_xps = m_xpn

            self.log.debug('Model decrease = %9.2e' % m_xps)

            # Compute trial point and evaluate local data.
            x_trial = x + step
            f_trial = model.obj(x_trial)
            c_trial = model.cons_pos(x_trial)
            theta_trial = 0.5 * np.dot(c_trial, c_trial)
            delta_f = -m_xps           # Overall improvement in the model.
            delta_ft = m_xpn - m_xps   # Improvement due to tangential step.
            Jspc = c + Jop * step

            # Compute improvement in linearized feasibility.
            delta_feas = theta - 0.5 * np.dot(Jspc, Jspc)

            # Decide whether to consider the current iteration
            # an f- or a c-iteration.

            if step_t_norm > 0 and (delta_f >= self.forcing(2, theta)) and \
                    delta_f >= kappa_delta * delta_ft and \
                    theta_trial <= theta_max:

                # Step 3. Consider that this is an f-iteration.
                it_type = 'f'

                # Decide whether trial point is accepted.
                ratio = (f - f_trial) / delta_f

                self.log.debug('f-iter ratio = %9.2e' % ratio)

                if ratio >= eta_1:    # Successful step.
                    suc = 's'
                    x = x_trial
                    f = f_trial
                    c = c_trial
                    self.step = step.copy()
                    theta = theta_trial

                    # Decide whether to update f-trust-region radius.
                    if ratio >= eta_2:
                        suc = 'v'
                        delta_f = min(max(delta_f, gamma_3 * step_norm),
                                      1.0e+10)

                    # Decide whether to update c-trust-region radius.
                    if theta_trial < eta_3 * theta_max:
                        ns = step_n_norm if step_n_norm > 0 else delta_c
                        delta_c = min(max(delta_c, gamma_3 * ns),
                                      1.0e+10)

                    self.log.debug('New delta_f = %8.2e' % delta_f)
                    self.log.debug('New delta_c = %8.2e' % delta_c)

                else:                 # Unsuccessful step (ratio < eta_1).

                    attempt_soc = True
                    suc = 'u'

                    if attempt_soc:

                        self.log.debug('  Attempting second-order correction')

                        # Attempt a second-order correction by solving
                        # minimize |c_trial + J n|  subject to  |n| <= delta_c.
                        step_soc, step_soc_norm, status_soc = \
                            self.lsq(Jop, -c_trial, radius=delta_c, reg=reg)

                        if status_soc != 'trust-region boundary active':

                            # Consider SOC step as candidate step.
                            x_soc = x_trial + step_soc
                            f_soc = model.obj(x_soc)
                            c_soc = model.cons_pos(x_soc)
                            theta_soc = 0.5 * np.dot(c_soc, c_soc)
                            ratio = (f - f_soc) / delta_f

                            # Decide whether to accept SOC step.
                            if ratio >= eta_1 and theta_soc <= theta_max:
                                suc = '2'
                                x = x_soc
                                f = f_soc
                                c = c_soc
                                theta = theta_soc
                                self.step = step + step_soc
                            else:

                                # Backtracking linesearch a la Nocedal & Yuan.
                                # Abandon SOC step. Backtrack from x+step.
                                if ny:
                                    (x, f, alpha) = self.nyf(x, f, f_trial,
                                                             g, step)
                                    # g = model.grad(x)
                                    c = model.cons_pos(x)
                                    theta = 0.5 * np.dot(c, c)
                                    self.step = step + alpha * step_soc
                                    delta_f = min(alpha, .8) * step_norm
                                    suc = 'y'

                                else:
                                    delta_f = gamma_1 * delta_f

                        else:  # SOC step lies on boundary of trust region.
                            delta_f = gamma_1 * delta_f

                    else:  # SOC step not attempted.

                        # Backtracking linesearch a la Nocedal & Yuan.
                        if ny:
                            (x, f, alpha) = self.nyf(x, f, f_trial, g, step)
                            c = model.cons_pos(x)
                            theta = 0.5 * np.dot(c, c)
                            self.step = alpha * step
                            delta_f = min(alpha, .8) * step_norm
                            suc = 'y'
                        else:
                            delta_f = gamma_1 * delta_f

            else:

                # Step 4. Consider that this is a c-iteration.
                it_type = 'c'

                # Display information.
                self.log.debug('c-iteration because ')
                if step_t_norm == 0.0:
                    self.log.debug('|t|=0')
                if delta_f < self.forcing(2, theta):
                    self.log.debug('delta_f=%8.2e < forcing=%8.2e' %
                                   (delta_f, self.forcing(2, theta)))
                if delta_f < kappa_delta * delta_ft:
                    self.log.debug('delta_f=%8.2e < frac * delta_ft=%8.2e' %
                                   (delta_f, delta_ft))
                if theta_trial > theta_max:
                    self.log.debug('theta_trial=%8.2e > theta_max=%8.2e' %
                                   (theta_trial, theta_max))

                # Step 4.1. Check trial point for acceptability.
                if delta_feas < 0:
                    self.log.debug(' !!! Warning: delta_feas is negative !!!')

                ratio = (theta - theta_trial + 1.e-16) / (delta_feas + 1.e-16)
                self.log.debug('c-iter ratio = %9.2e' % ratio)

                if ratio >= eta_1:  # Successful step.
                    x = x_trial
                    f = f_trial
                    c = c_trial
                    self.step = step.copy()
                    suc = 's'

                    # Step 4.2. Update delta_c.
                    if ratio >= eta_2:     # Very successful step.
                        ns = step_n_norm if step_n_norm > 0 else delta_c
                        delta_c = min(max(delta_c, gamma_3 * ns),
                                      1.0e+10)
                        suc = 'v'

                    # Step 4.3. Update maximum acceptable infeasibility.
                    theta_max = max(kappa_tx1 * theta_max,
                                    kappa_tx2 * theta +
                                    (1 - kappa_tx2) * theta_trial)
                    theta = theta_trial

                else:                      # Unsuccessful step.

                    # Backtracking linesearch a la Nocedal & Yuan.
                    ns = step_n_norm if step_n_norm > 0 else delta_c
                    if ny:
                        (x, c, theta, alpha) = self.nyc(x, theta, theta_trial,
                                                        c, Jop.T * c, step)
                        f = model.obj(x)
                        self.step = alpha * step
                        delta_c = min(alpha, .8) * ns
                        suc = 'y'
                    else:
                        delta_c = gamma_1 * ns
                        suc = 'u'

                self.log.debug('New delta_c = %8.2e' % delta_c)
                self.log.debug('New theta_max = %8.2e' % theta_max)

            # Step 5. Book keeping.
            if ratio >= eta_1 or ny:
                g = model.grad(x)
                if self.save_J:
                    self.J_old = J.copy()
                    J = model.jac(x)
                    self.J = J
                else:
                    J = model.jac(x)
                Jop = model.jop(x)
                self.post_iteration()

                p_norm = c_norm = 0
                if m > 0:
                    p_norm = np.linalg.norm(c)
                    c_norm = np.linalg.norm(c, np.inf)
                    grad_lag = g + Jop.T * y
                else:
                    grad_lag = g.copy()
                d_norm = np.linalg.norm(grad_lag) / (1 + np.linalg.norm(y))

            if self.it % 20 == 0:
                self.log.info(self.hdr)
            self.log.info(self.linefmt % (self.it, it_type, suc, status_n,
                                          status_t, f, p_norm, d_norm, delta_f,
                                          delta_c, theta_max, cgiter))

            optimal = (p_norm <= stop_p) and (d_norm <= self.stop_d)
        # End while.

        self.tsolve = cputime() - tsolve

        if optimal:
            self.status = 0  # Successful solve.
            self.log.info('Found an optimal solution! Yeah!')
        else:
            self.status = 1  # Refine this in the future.

        self.x = x
        self.f = f
        self.optimal = optimal
        self.p_resid = p_norm
        self.d_resid = d_norm
        self.niter = self.it

        return


class LSTRFunnel(Funnel):
    """A Variant of the Funnel class using LSTR instead of LSQR."""

    def lsq(self, A, b, radius=None, **kwargs):
        """Solve the linear least-squares problem in the variable x.

            minimize |Ax - b|

        in the Euclidian norm with LSTR. Optionally, x may be
        subject to a Euclidian-norm trust-region constraint
        |x| <= radius. This function returns (x, x_norm, status).
        """
        LSTR = LSTRFramework(A, -b, radius=radius)
        LSTR.Solve()
        return (LSTR.step, LSTR.stepNorm, LSTR.status)
