u"""Trust-Region Method for Bound-Constrained Programming.

A pure Python/Numpy implementation of TRON as described in

    Chih-Jen Lin and Jorge J. Moré, *Newton's Method for Large Bound-
    Constrained Optimization Problems*, SIAM J. Optim., 9(4), 1100–1127, 1999.
"""
import numpy as np
import logging
from pykrylov.linop import SymmetricallyReducedLinearOperator as ReducedHessian
from nlp.model.nlpmodel import NLPModel
from nlp.tr.trustregion import GeneralizedTrustRegion
from nlp.tools import norms
from nlp.tools.utils import where
from nlp.tools.timing import cputime
from nlp.tools.exceptions import UserExitRequest


__docformat__ = 'restructuredtext'


class TRON(object):
    """Trust-region Newton method for bound-constrained optimization problems.

           min f(x)  subject to xl <= x <= xu

    where the Hessian matrix is sparse.
    """

    def __init__(self, model, **kwargs):
        """Instantiate a trust-region solver for ``model``.

        :parameters:
            :model:        a :class:`NLPModel` instance.

        :keywords:
            :x0:           starting point                     (``model.x0``)
            :reltol:       relative stopping tolerance        (1.0e-12)
            :abstol:       absolute stopping tolerance        (1.0e-6)
            :maxiter:      maximum number of iterations       (max(1000,10n))
            :maxfuncall:   maximum number of objective function evaluations
                                                              (1000)
            :logger_name:  name of a logger object that can be used in the post
                           iteration                          (``None``)
        """
        if isinstance(model, NLPModel):
            self.model = model
        else:
            raise TypeError("Model supplied is not a subclass of `NLPModel`.")
        self.TR = GeneralizedTrustRegion()
        self.iter = 0         # Iteration counter
        self.total_cgiter = 0
        self.x = kwargs.get('x0', self.model.x0.copy())
        self.f = None
        self.f0 = None
        self.g = None
        self.g_old = None
        self.save_g = False
        self.gnorm = None
        self.g0 = None
        self.tsolve = 0.0

        self.reltol = kwargs.get('reltol', 1e-12)
        self.abstol = kwargs.get('abstol', 1e-6)
        self.maxiter = kwargs.get('maxiter', 100 * self.model.n)
        self.maxfuncall = kwargs.get('maxfuncall', 1000)
        self.cgtol = 0.1
        self.gtol = 1.0e-5
        self.alphac = 1
        self.feval = 0

        self.hformat = '%-5s  %8s  %7s  %5s  %8s  %8s  %8s  %4s'
        self.header = self.hformat % ('Iter', 'f(x)', '|g(x)|', 'cg',
                                      'rho', 'Step', 'Radius', 'Stat')
        self.hlen = len(self.header)
        self.hline = '-' * self.hlen
        self.format = '%-5d  %8.2e  %7.1e  %5d  %8.1e  %8.1e  %8.1e  %4s'
        self.format0 = '%-5d  %8.2e  %7.1e  %5s  %8s  %8s  %8.1e  %4s'
        self.radii = [self.TR.radius]

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlp.tron')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False

    def precon(self, v, **kwargs):
        """Generic preconditioning method---must be overridden."""
        return v

    def post_iteration(self, **kwargs):
        """Override this method to perform work at the end of an iteration.

        For example, use this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def project(self, x, xl, xu):
        """Project x into the bounds [xl, xu]."""
        return np.maximum(np.minimum(x, xu), xl)

    def projected_step(self, x, d, xl, xu):
        """Compute the projected step x + d into the feasible box.

        Feasible box is defined as

                   xl <= x <= xu

                   s = P[x + d] - x
        """
        return self.project(x + d, xl, xu) - x

    def breakpoints(self, x, d, xl, xu):
        """Find the smallest and largest breakpoints on the half line x + t d.

        We assume that x is feasible. Return the smallest and largest t such
        that x + t d lies on the boundary.
        """
        pos = where((d > 0) & (x < xu))  # Hit the upper bound.
        neg = where((d < 0) & (x > xl))  # Hit the lower bound.
        npos = len(pos)
        nneg = len(neg)

        nbrpt = npos + nneg
        # Handle the exceptional case.
        if nbrpt == 0:
            return (0, 0, 0)

        brptmin = np.inf
        brptmax = 0
        if npos > 0:
            steps = (xu[pos] - x[pos]) / d[pos]
            brptmin = min(brptmin, np.min(steps))
            brptmax = max(brptmax, np.max(steps))
        if nneg > 0:
            steps = (xl[neg] - x[neg]) / d[neg]
            brptmin = min(brptmin, np.min(steps))
            brptmax = max(brptmax, np.max(steps))

        self.log.debug('Nearest  breakpoint: %7.1e', brptmin)
        self.log.debug('Farthest breakpoint: %7.1e', brptmax)
        return (nbrpt, brptmin, brptmax)

    def cauchy(self, x, g, H, xl, xu, delta, alpha):
        u"""Compute a Cauchy step.

        This step must satisfy a trust region constraint and a sufficient
        decrease condition.

        The Cauchy step is computed for the quadratic

           q(s) = 0.5 s'Hs + g's,

        where H is a symmetric matrix and g is a vector.
        Given a parameter alpha, the Cauchy step is

           s[α] = P[x - α g] - x,

        with P the projection onto the n-dimensional interval [xl,xu].
        The Cauchy step satisfies the trust region constraint and the
        sufficient decrease condition

           || s || <= Δ,      q(s) <= mu_0 (g's),

        where mu_0 is a constant in (0, 1).
        """
        # Constant that defines sufficient decrease.
        mu0 = 0.01
        # Interpolation and extrapolation factors.
        interpf = 0.1
        extrapf = 10

        # Find the minimal and maximal break-point on x - alpha*g.
        (_, _, brptmax) = self.breakpoints(x, -g, xl, xu)

        # Evaluate the initial alpha and decide if the algorithm
        # must interpolate or extrapolate.
        s = self.projected_step(x, -alpha * g, xl, xu)
        if norms.norm2(s) > delta:
            interp = True
        else:
            Hs = H * s
            gts = np.dot(g, s)
            interp = (.5 * np.dot(Hs, s) + gts >= mu0 * gts)

        # Either interpolate or extrapolate to find a successful step.
        if interp:
            # Reduce alpha until a successful step is found.
            search = True
            while search:
                alpha = interpf * alpha
                s = self.projected_step(x, -alpha * g, xl, xu)
                if norms.norm2(s) <= delta:
                    Hs = H * s
                    gts = np.dot(g, s)
                    search = (.5 * np.dot(Hs, s) + gts >= mu0 * gts)
        else:
            # Increase alpha until a successful step is found.
            search = True
            alphas = alpha
            while search and alpha <= brptmax:
                alpha = extrapf * alpha
                s = self.projected_step(x, -alpha * g, xl, xu)
                if norms.norm2(s) <= delta:
                    Hs = H * s
                    gts = np.dot(g, s)
                    if .5 * np.dot(Hs, s) + gts < mu0 * gts:
                        search = True
                        alphas = alpha
                else:
                    search = False

            # Recover the last successful step.
            alpha = alphas
            s = self.projected_step(x, -alpha * g, xl, xu)
        return (s, alpha)

    def trqsol(self, x, p, delta):
        u"""Compute a solution of the quadratic trust region equation.

        It returns the largest (non-negative) solution of
            ||x + σ p|| = Δ.

        The code is only guaranteed to produce a non-negative solution
        if ||x|| <= Δ, and p != 0.
        If the trust region equation has no solution, σ is set to 0.
        """
        ptx = np.dot(p, x)
        ptp = np.dot(p, p)
        xtx = np.dot(x, x)
        dsq = delta**2

        # Guard against abnormal cases.
        rad = ptx**2 + ptp * (dsq - xtx)
        rad = np.sqrt(max(rad, 0))

        if ptx > 0:
            sigma = (dsq - xtx) / (ptx + rad)
        elif rad > 0:
            sigma = (rad - ptx) / ptp
        else:
            sigma = 0
        return sigma

    def truncated_cg(self, g, H, delta, tol, itermax):
        u"""Preconditioned conjugate-gradient method.

        This method uses a preconditioned conjugate gradient method
        to find an approximate minimizer of the trust region subproblem

           min { q(s) : || s || <= Δ }.

        where q is the quadratic

           q(s) = 0.5 s'Hs + g's,

        and H is a symmetric matrix.

        Returned status is one of the following:
            info = 1  Convergence test is satisfied.
                      || ∇ q(s) || <= tol

            info = 2  Failure to converge within itermax iterations.

            info = 3  Conjugate gradient iterates leave the trust-region.

            info = 4  The trust-region bound does not allow further progress.
        """
        # Initialize the iterate w and the residual r.
        w = np.zeros(len(g))

        # Initialize the residual r of grad q to -g.
        r = -g.copy()

        # Initialize the direction p.
        p = r.copy()

        # Initialize rho and the norm of r.
        rho = np.dot(r, r)
        rnorm0 = np.sqrt(rho)

        # Exit if g = 0.
        if rnorm0 == 0:
            iters = 0
            info = 1
            return (w, iters, info)

        for iters in range(0, itermax):
            z = p.copy()
            q = H * z

            # Compute alpha and determine sigma such that the trust region
            # constraint || w + sigma*p || = delta is satisfied.
            ptq = np.dot(p, q)
            if ptq > 0:
                alpha = rho / ptq
            else:
                alpha = 0

            sigma = self.trqsol(w, p, delta)

            # Exit if there is negative curvature or if the
            # iterates exit the trust region.
            if ptq <= 0 or alpha >= sigma:
                w = sigma * p + w
                if ptq <= 0:
                    info = 3
                else:
                    info = 4
                return (w, iters, info)

            # Update w and the residual r
            w = alpha * p + w
            r = -alpha * q + r

            # Exit if the residual convergence test is satisfied.
            rtr = np.dot(r, r)
            rnorm = np.sqrt(rtr)
            if rnorm <= tol:
                info = 1
                return (w, iters, info)

            # Compute p = r + beta*p and update rho.
            beta = rtr / rho
            p = r + p * beta
            rho = rtr

        # iters = itmax
        info = 2

        return (w, iters, info)

    def projected_newton_step(self, x, g, H, delta, xl, xu, s, cgtol, itermax):
        """Generate a sequence of approximate minimizers to the QP subprolem.

            min q(x) subject to  xl <= x <= xu

        where q(x[0] + s) = 0.5 s' H s + g' s,

        x[0] is a base point provided by the user, H is a symmetric
        matrix and g is a vector.

        At each stage we have an approximate minimizer x[k], and generate
        a direction p[k] by using a preconditioned conjugate gradient
        method on the subproblem

           min { q(x[k] + p) : || p || <= delta, s(fixed) = 0 },

        where fixed is the set of variables fixed at x[k] and delta is the
        trust region bound. Given p[k],
        the next minimizer x[k+1] is generated by a projected search.

        The starting point for this subroutine is x[1] = x[0] + s, where
        x[0] is a base point and s is the Cauchy step.

        Returned status is one of the following:
            info = 1  Convergence. The final step s satisfies
                      || (g + H s)[free] || <= cgtol || g[free] ||, and the
                      final x is an approximate minimizer in the face defined
                      by the free variables.

            info = 2  Termination. The trust region bound does not allow
                      further progress: || p[k] || = delta.

            info = 3  Failure to converge within itermax iterations.
        """
        w = H * s

        # Compute the Cauchy point.
        x = self.project(x + s, xl, xu)

        # Start the main iteration loop.
        # There are at most n iterations because at each iteration
        # at least one variable becomes active.
        iters = 0
        for i in range(0, len(x)):
            # Determine the free variables at the current minimizer.
            free_vars = where((x > xl) & (x < xu))
            nfree = len(free_vars)

            # Exit if there are no free constraints.
            if nfree == 0:
                info = 1
                return (x, s, iters, info)

            # Obtain the submatrix of H for the free variables.
            ZHZ = ReducedHessian(H, free_vars)

            # Compute the norm of the reduced gradient Z'*g
            gfree = g[free_vars] + w[free_vars]
            gfnorm = norms.norm2(g[free_vars])

            # Solve the trust region subproblem in the free variables
            # to generate a direction p[k]

            tol = cgtol * gfnorm
            (w, trpcg_iters, infotr) = self.truncated_cg(gfree, ZHZ, delta,
                                                         tol, 1000)
            iters += trpcg_iters

            # Use a projected search to obtain the next iterate
            xfree = x[free_vars]
            xlfree = xl[free_vars]
            xufree = xu[free_vars]
            (xfree, w) = self.projected_linesearch(xfree, xlfree, xufree,
                                                   gfree, w, ZHZ, alpha=1.0)

            # Update the minimizer and the step.
            # Note that s now contains x[k+1] - x[0]
            x[free_vars] = xfree
            s[free_vars] = s[free_vars] + w

            # Compute the gradient grad q(x[k+1]) = g + H*(x[k+1] - x[0])
            # of q at x[k+1] for the free variables.
            w = H * s
            gfree = g[free_vars] + w[free_vars]
            gfnormf = norms.norm2(gfree)

            # Convergence and termination test.
            # We terminate if the preconditioned conjugate gradient method
            # encounters a direction of negative curvature, or
            # if the step is at the trust region bound.
            if gfnormf <= cgtol * gfnorm:
                info = 1
                return (x, s, iters, info)
            elif infotr == 3 or infotr == 4:
                info = 2
                return (x, s, iters, info)
            elif iters >= itermax:
                info = 3
                return (x, s, iters, info)

        return (x, s, iters, info)

    def projected_linesearch(self, x, xl, xu, g, d, H, alpha=1.0):
        u"""Use a projected search to compute a satisfactory step.

        This step must satisfy a sufficient decrease condition for the
        quadratic

            q(s) = 0.5 s'Hs + g's,

        where H is a symmetric matrix and g is a vector.
        Given the parameter α, the step is

           s[α] = P[x + α d] - x,

        where d is the search direction and P the projection onto the
        n-dimensional interval [xl,xu]. The final step s = s[α] satisfies
        the sufficient decrease condition

           q(s) <= mu_0*(g'*s),

        where mu_0 is a constant in (0, 1).

        The search direction d must be a descent direction for the quadratic q
        at x such that the quadratic is decreasing in the ray  x + α d
        for 0 <= α <= 1.
        """
        mu0 = 0.01
        interpf = 0.5
        nsteps = 0

        # Find the smallest break-point on the ray x + alpha*d.
        (_, brptmin, _) = self.breakpoints(x, d, xl, xu)

        # Reduce alpha until the sufficient decrease condition is
        # satisfied or x + alpha*w is feasible.

        search = True
        while search and alpha > brptmin:

            # Calculate P[x + alpha*w] - x and check the sufficient
            # decrease condition.
            nsteps += 1

            s = self.projected_step(x, alpha * d, xl, xu)
            Hs = H * s
            gts = np.dot(g, s)
            q = .5 * np.dot(Hs, s) + gts
            if q <= mu0 * gts:
                search = False
            else:
                alpha = interpf * alpha

        # Force at least one more constraint to be added to the active
        # set if alpha < brptmin and the full step is not successful.
        # There is sufficient decrease because the quadratic function
        # is decreasing in the ray x + alpha*w for 0 <= alpha <= 1.
        if alpha < 1 and alpha < brptmin:
            alpha = brptmin

        # Compute the final iterate and step.
        s = self.projected_step(x, alpha * d, xl, xu)
        x = self.project(x + alpha * s, xl, xu)
        return (x, s)

    def projected_gradient_norm2(self, x, g, xl, xu):
        """Compute the Euclidean norm of the projected gradient at x."""
        lower = where(x == xl)
        upper = where(x == xu)

        pg = g.copy()
        pg[lower] = np.minimum(g[lower], 0)
        pg[upper] = np.maximum(g[upper], 0)

        return norms.norm2(pg[where(xl != xu)])

    def solve(self):
        """Solve method.

        All keyword arguments are passed directly to the constructor of the
        trust-region solver.
        """
        model = self.model

        # Project the initial point into [xl,xu].
        self.project(self.x, model.Lvar, model.Uvar)

        # Gather initial information.
        self.f = model.obj(self.x)
        self.feval += 1
        self.f0 = self.f
        self.g = model.grad(self.x)  # Current  gradient
        self.g_old = self.g.copy()
        self.gnorm = norms.norm2(self.g)
        self.g0 = self.gnorm
        cgtol = self.cgtol
        cg_iter = 0
        cgitermax = model.n

        # Initialize the trust region radius
        self.TR.radius = self.g0

        # Test for convergence or termination
        stoptol = self.gtol * self.g0
        exitUser = False
        exitOptimal = False
        exitIter = self.iter >= self.maxiter
        exitFunCall = self.feval >= self.maxfuncall
        status = ''

        # Wrap Hessian into an operator.
        H = model.hop(self.x, self.model.pi0)
        t = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0:
            self.log.info(self.hline)
            self.log.info(self.header)
            self.log.info(self.hline)
            self.log.info(self.format0, self.iter, self.f, self.gnorm,
                          '', '', '', self.TR.radius, '')

        while not (exitUser or exitOptimal or exitIter or exitFunCall):
            self.iter += 1

            # Compute a step and evaluate the function at the trial point.

            # Save the best function value, iterate, and gradient.
            self.fc = self.f
            self.xc = self.x.copy()
            self.gc = self.g.copy()

            # Compute the Cauchy step and store in s.
            (s, self.alphac) = self.cauchy(self.x, self.g, H,
                                           model.Lvar, model.Uvar,
                                           self.TR.radius,
                                           self.alphac)

            # Compute the projected Newton step.
            (x, s, cg_iter, info) = self.projected_newton_step(self.x, self.g,
                                                               H,
                                                               self.TR.radius,
                                                               model.Lvar,
                                                               model.Uvar, s,
                                                               cgtol,
                                                               cgitermax)

            snorm = norms.norm2(s)
            self.total_cgiter += cg_iter

            # Compute the predicted reduction.
            m = np.dot(s, self.g) + .5 * np.dot(s, H * s)

            # Compute the function
            x_trial = self.x + s
            f_trial = model.obj(x_trial)
            self.feval += 1

            # Evaluate the step and determine if the step is successful.

            # Compute the actual reduction.
            rho = self.TR.ratio(self.f, f_trial, m)
            ared = self.f - f_trial

            # On the first iteration, adjust the initial step bound.
            snorm = norms.norm2(s)
            if self.iter == 1:
                self.TR.radius = min(self.TR.radius, snorm)

            # Update the trust region bound
            slope = np.dot(self.g, s)
            if f_trial - self.f - slope <= 0:
                alpha = self.TR.gamma3
            else:
                alpha = max(self.TR.gamma1,
                            -0.5 * (slope / (f_trial - self.f - slope)))

            # Update the trust region bound according to the ratio
            # of actual to predicted reduction
            self.TR.update_radius(rho, snorm, alpha)

            # Update the iterate.
            if rho > self.TR.eta0:
                # Successful iterate
                # Trust-region step is accepted.
                self.x = x_trial
                self.f = f_trial
                self.g = model.grad(self.x)
                self.gnorm = norms.norm2(self.g)
                step_status = 'Acc'

            else:
                # Unsuccessful iterate
                # Trust-region step is rejected.
                step_status = 'Rej'

            self.step_status = step_status
            self.radii.append(self.TR.radius)
            status = ''
            try:
                self.post_iteration()
            except UserExitRequest:
                status = 'usr'

            # Print out header, say, every 20 iterations
            if self.iter % 20 == 0:
                self.log.info(self.hline)
                self.log.info(self.header)
                self.log.info(self.hline)

            pstatus = step_status if step_status != 'Acc' else ''
            self.log.info(self.format, self.iter, self.f, self.gnorm,
                          cg_iter, rho, snorm, self.TR.radius, pstatus)

            # Test for convergence. FATOL and FRTOL
            if abs(ared) <= self.abstol and -m <= self.abstol:
                exitOptimal = True
                status = 'fatol'
            if abs(ared) <= self.reltol * abs(self.f) and \
               (-m <= self.reltol * abs(self.f)):
                exitOptimal = True
                status = 'frtol'

            if pstatus == '':
                pgnorm2 = self.projected_gradient_norm2(self.x, self.g,
                                                        model.Lvar, model.Uvar)
                if pgnorm2 <= stoptol:
                    exitOptimal = True
                    status = 'gtol'
            else:
                self.iter -= 1  # to match TRON iteration number

            exitIter = self.iter > self.maxiter
            exitFunCall = self.feval >= self.maxfuncall
            exitUser = status == 'usr'

        self.tsolve = cputime() - t    # Solve time
        # Set final solver status.
        if status == 'usr':
            pass
        elif self.iter > self.maxiter:
            status = 'itr'
        self.status = status
