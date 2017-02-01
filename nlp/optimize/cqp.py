# -*- coding: utf-8 -*-
"""Long-step primal-dual interior-point method for convex QP.

From Algorithm IPF on p.110 of Stephen J. Wright's book
"Primal-Dual Interior-Point Methods", SIAM ed., 1997.
The method uses the augmented system formulation. These systems are solved
using MA27 or MA57.

D. Orban, Montreal 2009-2011.
"""
try:                            # To solve augmented systems
    from hsl.solvers.pyma57 import PyMa57Solver as LBLContext
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLContext
from hsl.scaling.mc29 import mc29ad
from nlp.model.pysparsemodel import PySparseSlackModel
from pysparse.sparse import PysparseMatrix
from nlp.tools.norms import norm2, norm_infty, normest
from nlp.tools.timing import cputime
import logging
import numpy as np


class RegQPInteriorPointSolver(object):
    u"""Solve a QP with the primal-dual-regularized interior-point method.

    Solve a convex quadratic program of the form::

       minimize    q + cᵀx + ½ xᵀ Q x
       subject to  Ax = b                                  (QP)
                   l ≤ x ≤ u

    where Q is a symmetric positive semi-definite matrix. Any
    quadratic program may be converted to the above form by instantiation
    of the `SlackModel` class. The conversion to the slack formulation
    is mandatory in this implementation.

    The method is a variant of Mehrotra's predictor-corrector method where
    steps are computed by solving the primal-dual system in augmented form.
    A long-step variant is also available.

    Primal and dual regularization parameters may be specified by the user
    via the opional keyword arguments `primal_reg` and `dual_reg`. Both should be
    positive real numbers and should not be "too large". By default they
    are set to 1.0 and updated at each iteration.

    Problem scaling options are provided through the `scale_type` key word.

    Advantages of this method are that it is not sensitive to dense columns
    in A, no special treatment of the unbounded variables x is required,
    and a sparse symmetric quasi-definite system of equations is solved at
    each iteration. The latter, although indefinite, possesses a
    Cholesky-like factorization. Those properties make the method
    typically more robust than a standard predictor-corrector
    implementation and the linear system solves are often much faster than
    in a traditional interior-point method in augmented form.
    """

    def __init__(self, qp, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix [Aᴱ 0 ; Aᴵ -I] prior to solution
                         (default: `none`).

            :primal_reg_init: Initial value of primal regularization parameter
                              (default: `1.0`).

            :primal_reg_min: Minimum value of primal regularization parameter
                             (default: `1.0e-8`).

            :dual_reg_init: Initial value of dual regularization parameter
                            (default: `1.0`).

            :dual_reg_min: Minimum value of dual regularization parameter
                           (default: `1.0e-8`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :estimate_cond: Estimate the matrix condition number when solving
                            the linear system (default: `False`)

            :itermax: Max number of iterations. (default: max(100,10*qp.n))

            :mehrotra_pc: Use Mehrotra's predictor-corrector strategy
                          to update the step (default: `True`) If `False`,
                          use a variant of the long-step method.

            :stoptol: The convergence tolerance (default: 1.0e-6)

            :check_infeas: Check for an infeasible problem (default: `True`)
        """
        if not isinstance(qp, PySparseSlackModel):
            msg = 'The QP model must be an instance of SlackModel with sparse'
            msg2 = ' Hessian and Jacobian matrices available.'
            raise TypeError(msg+msg2)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'cqp.solver')
        self.log = logging.getLogger(logger_name)

        # Either none, abs, or mc29
        self.scale_type = kwargs.get('scale_type', 'none')

        self.qp = qp

        # Solver cannot support QPs with fixed variables at this time
        if qp.nfixedB > 0:
            msg = 'The QP model must not contain fixed variables'
            raise ValueError(msg)

        # Compute the size of the linear system in the problem,
        # i.e., count the number of finite bounds and fixed variables
        # present in the problem to determine the true size
        self.n = qp.n
        self.m = qp.m
        self.nl = qp.nlowerB + qp.nrangeB
        self.nu = qp.nupperB + qp.nrangeB
        self.sys_size = self.n + self.m + self.nl + self.nu

        # Some useful index lists for associating variables with bound
        # multipliers
        self.all_lb = qp.lowerB + qp.rangeB
        self.all_lb.sort()
        self.all_ub = qp.upperB + qp.rangeB
        self.all_ub.sort()

        # Compute indices of the range variables within the all_lb and
        # all_ub arrays (used in the initial point calculation)
        self.range_in_lb = []
        self.range_in_ub = []
        for k in qp.rangeB:
            self.range_in_lb.append(self.all_lb.index(k))
            self.range_in_ub.append(self.all_ub.index(k))

        # Collect basic info about the problem.
        zero_pt = np.zeros(qp.n)
        self.q = qp.obj(zero_pt)    # Constant term in the objective
        self.b = -qp.cons(zero_pt)  # Constant term in the constraints
        self.c = qp.grad(zero_pt)   # Vector term in the objective
        self.A = qp.jac(zero_pt)    # Jacobian including slack blocks
        self.Q = qp.hess(zero_pt)   # Hessian including slack blocks

        # A few useful norms to measure algorithm convergence
        self.normb = norm2(self.b)
        self.normc = norm2(self.c)
        self.normA = self.A.matrix.norm('fro')
        self.normQ = self.Q.matrix.norm('fro')

        # Initialize the augmented system matrix
        # This matrix object is used both in calculating the starting point
        # and the steps in the algorithm. Blocks common to all problems,
        # i.e., the sparse Q and A matrices, are also put in place
        self.H = PysparseMatrix(size=self.sys_size,
            sizeHint=2*self.nl + 2*self.nu + self.A.nnz + self.Q.nnz,
            symmetric=True)
        self.H[:n, :n] = self.Q
        self.H[n:n+m, :n] = self.A

        # ** DEV NOTE: Moved scaling call to solve function **

        # It will be more efficient to keep the diagonal of Q around.
        self.diagQ = self.Q.take(range(self.n))

        # We perform the analyze phase on the augmented system only once.
        # self.LBL will be initialized in solve().
        self.LBL = None

        # Set regularization parameters.
        self.primal_reg_init = kwargs.get('primal_reg_init', 1.0)
        self.primal_reg_min = kwargs.get('primal_reg_min',1.0e-8)
        self.dual_reg_init = kwargs.get('dual_reg_init', 1.0)
        self.dual_reg_min = kwargs.get('dual_reg_min',1.0e-8)

        # Check input regularization parameters.
        if self.primal_reg_init < 0.0 or self.dual_reg_init < 0.0:
            raise ValueError('Regularization parameters must be nonnegative.')
        if self.primal_reg_min < 0.0 or self.dual_reg_min < 0.0:
            raise ValueError('Minimum regularization parameters must be nonnegative.')

        # Set working regularization values
        if self.primal_reg_init < self.primal_reg_min:
            self.primal_reg = self.primal_reg_min
        else:
            self.primal_reg = self.primal_reg_init

        if self.dual_reg_init < self.dual_reg_min:
            self.dual_reg = self.dual_reg_min
        else:
            self.dual_reg = self.dual_reg_init

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 5)

        # Parameters for the LBL solve and iterative refinement
        self.itref_threshold = 1.e-5
        self.nitref = 5

        # ** DEV NOTE: This is WAY TOO MUCH to display on one line **
        # ** to be trimmed later **

        # Initialize format strings for display
        fmt_hdr = '%-4s  %9s' + '  %-8s' * 6 + \
                  '  %-7s  %-4s  %-4s' + '  %-8s' * 8
        self.header = fmt_hdr % ('Iter', 'Cost', 'pResid', 'dResid', 'cResid',
                                 'rGap', 'qNorm', 'rNorm', 'Mu', 'AlPr',
                                 'AlDu', 'LS Resid', 'RegPr', 'RegDu', 'Rho q',
                                 'Del r', 'Min(s)', 'Min(z)', 'Max(s)')
        self.format1 = '%-4d  %9.2e'
        self.format1 += '  %-8.2e' * 6
        self.format2 = '  %-7.1e  %-4.2f  %-4.2f'
        self.format2 += '  %-8.2e' * 8

        # Additional options to collect
        self.estimate_cond = kwargs.get('estimate_cond', False)
        self.itermax = kwargs.get('itermax', max(100, 10*qp.n))
        self.stoptol = kwargs.get('stoptol', 1.0e-6)
        self.mehrotra_pc = kwargs.get('mehrotra_pc', True)

        return

    def scale(self):
        """Compute scaling factors for the problem.

        If the solver is run with scaling, this function computes scaling
        factors for the rows and columns of the Jacobian so that the scaled
        matrix entries have some nicer properties.

        In effect the original problem::

            minimize    c' x + 1/2 x' Q x
            subject to  Ax = b
                        (bounds)

        is converted to::

            minimize    (Cc)' x + 1/2 x' (CQC') x
            subject to  R A C x = R b
                        (bounds)

        where the diagonal matrices R and C contain row and column scaling
        factors respectively.

        The options for the :scale_type: keyword are as follows:
            :none:  Do not calculate scaling factors (The resulting 
                    variables are of type `None`)

            :abs:   Compute each row scaling as the largest absolute-value
                    row entry, then compute each column scaling as the largest
                    absolute-value column entry.

            :mc29:  Compute row and column scaling using the Harwell
                    Subroutine MC29, available from the HSL.py interface.

        """
        log = self.log
        m, n = self.A.shape
        self.row_scale = np.zeros(m)
        self.col_scale = np.zeros(n)
        (values, irow, jcol) = self.A.find()

        if self.scale_type == 'none':

            self.row_scale = None
            self.col_scale = None

        elif self.scale_type == 'abs':

            log.debug('Smallest and largest elements of A prior to scaling: ')
            log.debug('%8.2e %8.2e' % (np.min(np.abs(values)),
                                          np.max(np.abs(values))))

            # Find row scaling.
            for k in range(len(values)):
                row = irow[k]
                val = abs(values[k])
                self.row_scale[row] = min(self.row_scale[row], 1/val)
            self.row_scale[row_scale == 0.0] = 1.0

            log.debug('Max row scaling factor = %8.2e' % np.max(self.row_scale))

            # Modified A values after row scaling
            temp_values = values * self.row_scale[irow]

            # Find column scaling.
            for k in range(len(temp_values)):
                col = jcol[k]
                val = abs(temp_values[k])
                self.col_scale[col] = max(self.col_scale[col], 1/val)
            self.col_scale[self.col_scale == 0.0] = 1.0

            log.debug('Max column scaling factor = %8.2e' % np.max(self.col_scale))

        elif self.scale_type == 'mc29':

            row_scale, col_scale, ifail = mc29ad(m, n, values, irow, jcol)

            # row_scale and col_scale contain in fact the logarithms of the
            # scaling factors. Modify these before storage
            self.row_scale = np.exp(row_scale)
            self.col_scale = np.exp(col_scale)

        else:

            log.info('Scaling option not recognized, no scaling will be applied.')
            self.row_scale = None
            self.col_scale = None

        return

    def solve(self, **kwargs):
        """Solve the problem.

        :returns:

            :x:            final iterate
            :y:            final value of the Lagrange multipliers associated
                           to `Ax = b`
            :zL:           final value of lower-bound multipliers
            :zU:           final value of upper-bound multipliers
            :obj_value:    final cost
            :iter:         total number of iterations
            :kktRes:     final relative residual
            :solve_time:   time to solve the QP
            :status:       string describing the exit status.
            :short_status: short version of status, used for printing.

        """
        qp = self.qp
        Lvar = qp.Lvar
        Uvar = qp.Uvar

        # Transfer pointers for convenience.
        A = self.A
        b = self.b
        c = self.c
        Q = self.Q
        nl = self.nl
        nu = self.nu

        # Obtain initial point from Mehrotra's heuristic.
        (self.x, self.y, self.zL, self.zU) = self.set_initial_guess()
        x = self.x
        y = self.y
        zL = self.zL
        zU = self.zU

        exitOpt = False
        exitInfeasP = False
        exitInfeasD = False
        exitIter = False
        exitReg = False
        iter = 0

        # Display header and initial point info
        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        output_line = self.format1 % (iter, cx + 0.5 * xQx, pResid,
                                      dResid, cResid, rgap, qNorm,
                                      rNorm)
        output_line += self.format2 % (mu, alpha_p, alpha_d,
                                       nres, self.primal_reg, self.dual_reg, rho_q,
                                       del_r, mins, minz, maxs)
        self.log.info(output_line)

        setup_time = cputime()

        # Main loop.
        while not (exitOpt or exitInfeasP or exitInfeasD or exitIter or exitReg):

            iter += 1

            # Display initial header every so often.
            if iter % 20 == 0:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))

            # Compute residuals.
            pFeas = A*x - b
            dFeas = Q*x + c - y*A - zL + zU
            lComp = zL*(x[self.all_lb] - Lvar[self.all_lb])
            uComp = zU*(Uvar[self.all_ub] - x[self.all_ub])

            # Compute complementarity measure.
            if (nl + nu) > 0:
                mu = (lComp.sum() + uComp.sum()) / (nl + nu)
            else:
                mu = 0.0

            # Compute residual norms and scaled residual norms.
            pFeasNorm = norm2(pFeas)
            pFeasNorm_scaled = pFeasNorm / (1 + self.normb + self.normA + self.normQ)
            dFeasNorm = norm2(dFeas)
            dFeasNorm_scaled = dFeasNorm / (1 + self.normc + self.normA + self.normQ)

            # Compute relative duality gap.
            dual_gap = mu / (1 + abs(np.dot(c,x)) + self.normA + self.normQ)

            # Compute overall residual for stopping condition.
            kktRes = max(pFeasNorm_scaled, dFeasNorm_scaled, dual_gap)

            # At the first iteration, initialize perturbation vectors
            # that regularize the problem.
            # Use r for the primal feasibility and s for the dual feasibility.
            # Also, initialize some parameters for infeasibility detection
            if iter == 1:
                s = -dFeas / self.primal_reg
                r = -pFeas / self.dual_reg
                pr_infeas_count = 0
                du_infeas_count = 0
                dFeasMin = dFeasNorm
                pFeasMin = pFeasNorm
                mu0 = mu

            else:

                self.dual_reg = max(self.dual_reg / 10, self.dual_reg_min)
                self.primal_reg = max(self.primal_reg / 10, self.primal_reg_min)

                # Check for primal infeasibility
                if mu < 0.01 * self.stoptol * mu0 and \
                        pFeasNorm > pFeasMin * (1.e-6 / self.stoptol) :
                    pr_infeas_count += 1
                    if pr_infeas_count > 6:
                        status = 'Problem seems to be (locally) dual'
                        status += ' infeasible'
                        short_status = 'dInf'
                        exitInfeasD = True
                        continue
                else:
                    pr_infeas_count = 0

                # Check for dual infeasibility
                if mu < 0.01 * self.stoptol * mu0 and \
                        dFeasNorm > dFeasMin * (1.e-6 / self.stoptol) :
                    du_infeas_count += 1
                    if du_infeas_count > 6:
                        status = 'Problem seems to be (locally) primal'
                        status += ' infeasible'
                        short_status = 'pInf'
                        exitInfeasP = True
                        continue
                else:
                    du_infeas_count = 0

            # Display objective and residual data.
            output_line = self.format1 % (iter, cx + 0.5 * xQx, pResid,
                                          dResid, cResid, rgap, qNorm,
                                          rNorm)

            if kktRes <= self.stoptol:
                status = 'Optimal solution found'
                short_status = 'opt'
                exitOpt = True
                continue

            if iter >= self.itermax:
                status = 'Maximum number of iterations reached'
                short_status = 'iter'
                exitIter = True
                continue

            # Compute augmented matrix and factorize it, checking for
            # degeneracy along the way
            self.set_system_matrix()
            self.LBL.factorize(self.H)

            if not self.LBL.isFullRank:
                nbump = 0
                while not self.LBL.isFullRank and nbump < self.bump_max:
                    self.log.debug('Primal-Dual Matrix Rank Deficient' +
                                      '... bumping up reg parameters')
                    nbump += 1
                    self.primal_reg *= 100
                    self.dual_reg *= 100
                    self.update_system_matrix()
                    self.LBL.factorize(self.H)

            # Exit if regularization is unsuccessful.
            if not self.LBL.isFullRank:
                status = 'Unable to regularize sufficiently.'
                short_status = 'degn'
                exitReg = True
                continue

            # Compute the right-hand side based on the step computation method
            if self.mehrotra_pc:
                # Compute affine-scaling step, i.e. with centering = 0.
                rhs_affine = np.zeros(self.sys_size)
                rhs_affine[:n] = -dFeas
                rhs_affine[n:n+m] = -pFeas
                rhs_affine[n+m:n+m+nl] = -(zL**0.5)*lComp
                rhs_affine[n+m+nl:] = (zU**0.5)*uComp

                self.solve_system(rhs_affine)

                dx_aff = self.LBL.x[:n]
                dy_aff = -self.LBL.x[n:n+m]
                dzL_aff = -(zL**0.5)*self.LBL.x[n+m:n+m+nl]
                dzU_aff = -(zU**0.5)*self.LBL.x[n+m:n+m+nl]

                # ** Carry on from here **

                # Compute largest allowed primal and dual stepsizes.
                (alpha_p, ip) = self.max_step_length(s, ds)
                (alpha_d, ip) = self.max_step_length(z, dz)

                # Estimate duality gap after affine-scaling step.
                muAff = np.dot(s + alpha_p * ds, z + alpha_d * dz) / ns
                sigma = (muAff / mu)**3

                # Incorporate predictor information for corrector step.
                # Only update rhs[on:n]; the rest of the vector did not change.
                comp += ds * dz
                comp -= sigma * mu
                self.update_corrector_rhs(rhs, s, z, comp)
            else:
                # Use long-step method: Compute centering parameter.
                sigma = min(0.1, 100 * mu)
                comp -= sigma * mu

                # Assemble rhs.
                self.update_long_step_rhs(rhs, pFeas, dFeas, comp, s)

            # Solve augmented system.
            (step, nres, neig) = self.solve_system(rhs)

            # Recover step.
            dx, ds, dy, dz = self.get_dxsyz(step, x, s, y, z, comp)

            # Compute largest allowed primal and dual stepsizes.
            (alpha_p, ip) = self.max_step_length(s, ds)
            (alpha_d, id) = self.max_step_length(z, dz)

            # Compute fraction-to-the-boundary factor.
            tau = max(.995, 1.0 - mu)

            if self.mehrotra_pc:
                # Compute actual stepsize using Mehrotra's heuristic.
                mult = 0.1

                # ip=-1 if ds ≥ 0, and id=-1 if dz ≥ 0
                if (ip != -1 or id != -1) and ip != id:
                    mu_tmp = np.dot(s + alpha_p * ds, z + alpha_d * dz) / ns

                if ip != -1 and ip != id:
                    zip = z[ip] + alpha_d * dz[ip]
                    gamma_p = ((mult * mu_tmp - s[ip] * zip) /
                               (alpha_p * ds[ip] * zip))
                    alpha_p *= max(1 - mult, gamma_p)

                if id != -1 and ip != id:
                    sid = s[id] + alpha_p * ds[id]
                    gamma_d = ((mult * mu_tmp - z[id] * sid) /
                               (alpha_d * dz[id] * sid))
                    alpha_d *= max(1 - mult, gamma_d)

                if ip == id and ip != -1:
                    # There is a division by zero in Mehrotra's heuristic
                    # Fall back on classical rule.
                    alpha_p *= tau
                    alpha_d *= tau

            else:
                alpha_p *= tau
                alpha_d *= tau

            # Display data.
            output_line += self.format2 % (mu, alpha_p, alpha_d,
                                           nres, primal_reg, dual_reg, rho_q,
                                           del_r, mins, minz, maxs)
            self.log.info(output_line)

            # Update iterates and perturbation vectors.
            x += alpha_p * dx    # This also updates slack variables.
            y += alpha_d * dy
            z += alpha_d * dz
            q *= (1 - alpha_p)
            q += alpha_p * dx
            r *= (1 - alpha_d)
            r += alpha_d * dy
            qNorm = norm2(q)
            rNorm = norm2(r)
            if primal_reg > 0:
                rho_q = primal_reg * qNorm / (1 + self.normc)
                rho_q_min = min(rho_q_min, rho_q)
            else:
                rho_q = 0.0
            if dual_reg > 0:
                del_r = dual_reg * rNorm / (1 + self.normb)
                del_r_min = min(del_r_min, del_r)
            else:
                del_r = 0.0
            iter += 1

        solve_time = cputime() - setup_time

        self.log.info('-' * len(self.header))

        # Transfer final values to class members.
        self.x = x
        self.y = y
        self.z = z
        self.iter = iter
        self.pResid = pResid
        self.cResid = cResid
        self.dResid = dResid
        self.rgap = rgap
        self.kktRes = kktRes
        self.solve_time = solve_time
        self.status = status
        self.short_status = short_status

        return

    def set_initial_guess(self):
        """Compute initial guess according the Mehrotra's heuristic.

        Initial values of x are computed as the solution to the
        least-squares problem::

            minimize    ½ xᵀQx + ½||rᴸ||² + ½||rᵁ||²
            subject to  Ax = b
                        rᴸ = x - l
                        rᵁ = u - x

        The solution is also the solution to the augmented system::

            [ Q   Aᵀ   I   I] [x ]   [0 ]
            [ A   0    0   0] [y']   [b ]
            [ I   0   -I   0] [rᴸ] = [l ]
            [ I   0    0  -I] [rᵁ]   [u ].

        Initial values for the multipliers y and z are chosen as the
        solution to the least-squares problem::

            minimize    ½ x'ᵀQx' + ½||zᴸ||² + ½||zᵁ||²
            subject to  Qx' + c - Aᵀy - zᴸ + zᵁ = 0
                        zᴸ = -(x - l)
                        zᵁ = -(u - x)

        which can be computed as the solution to the augmented system::

            [ Q   Aᵀ   I   I] [x']   [-c]
            [ A   0    0   0] [y ]   [ 0]
            [ I   0   -I   0] [zᴸ] = [ l]
            [ I   0    0  -I] [zᵁ]   [ u].

        To ensure stability and nonsingularity when A does not have full row
        rank or Q is singluar, the (1,1) block is perturbed by
        sqrt(self.primal_reg_min) * I and the (2,2) block is perturbed by
        sqrt(self.dual_reg_min) * I.

        The values of x', y', rᴸ, and rᵁ are discarded after solving the
        linear systems.

        The values of x and z are subsequently adjusted to ensure they
        are strictly within their bounds. See [Methrotra, 1992] for details.
        """
        qp = self.qp
        n = self.n
        m = self.m
        nl = self.nl
        nu = self.nu

        self.log.debug('Computing initial guess')

        # Set up augmented system matrix
        self.set_system_matrix(initial_guess=True)

        # Analyze and factorize the system
        self.LBL = LBLContext(self.H,
            sqd=(self.primal_reg_min > 0.0 and self.dual_reg_min > 0.0))

        # Assemble first right-hand side
        rhs_primal_init = np.zeros(self.sys_size)
        rhs_primal_init[n:n+m] = self.b
        rhs_primal_init[n+m:n+m+nl] = qp.Lvar[self.all_lb]
        rhs_primal_init[n+m+nl:] = qp.Uvar[self.all_ub]

        # Solve system and collect solution
        self.solve_system(rhs_primal_init)
        x_guess = self.LBL.x[:n].copy()

        # Assemble second right-hand side
        rhs_dual_init = np.zeros(self.sys_size)
        rhs_dual_init[:n] = -self.c
        rhs_dual_init[n+m:n+m+nl] = qp.Lvar[self.all_lb]
        rhs_dual_init[n+m+nl:] = qp.Uvar[self.all_ub]

        # Solve system and collect solution
        self.solve_system(rhs_dual_init)
        y = -self.LBL.x[n:n+m].copy()
        zL_guess = -self.LBL.x[n+m:n+m+nl].copy()
        zU_guess = self.LBL.x[n+m+nl:].copy()

        # Use Mehrotra's heuristic to compute a strictly feasible starting
        # point for all x and z
        rL_guess = x_guess[self.all_lb] - qp.Lvar[self.all_lb]
        rU_guess = qp.Uvar[self.all_ub] - x_guess[self.all_ub]

        drL = max(0.0, -1.5*np.min(rL_guess))
        drU = max(0.0, -1.5*np.min(rU_guess))
        dzL = max(0.0, -1.5*np.min(zL_guess))
        dzU = max(0.0, -1.5*np.min(zU_guess))

        rL_shift = drL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
            ((zL_guess + dzL).sum())
        zL_shift = dzL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
            ((rL_guess + drL).sum())
        rU_shift = drU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
            ((zU_guess + dzU).sum())
        zU_shift = dzU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
            ((rU_guess + drU).sum())

        rL = rL_guess + rL_shift
        rU = rU_guess + rU_shift
        zL = zL_guess + zL_shift
        zU = zU_guess + zU_shift

        x = x_guess
        x[self.all_lb] = qp.Lvar[self.all_lb] + rL
        x[self.all_ub] = qp.Uvar[self.all_ub] - rU

        # An additional normalization step for the range-bounded variables
        #
        # This normalization prevents the shift computed in rL and rU from
        # taking us outside the feasible range, and yields the same final
        # x value whether we take (Lvar + rL*norm) or (Uvar - rU*norm) as x
        intervals = qp.Uvar[qp.rangeB] - qp.Lvar[qp.rangeB]
        norm_factors = intervals / (intervals + rL_shift + rU_shift)
        x[qp.rangeB] = qp.Lvar[qp.rangeB] + rL[self.range_in_lb]*norm_factors

        # Check strict feasibility
        if not np.all(x > qp.Lvar) or not np.all(x < qp.Uvar) or \
        not np.all(zL > 0) or not np.all(zU > 0):
            raise ValueError('Initial point not strictly feasible')

        return (x, y, zL, zU)

    def max_step_length(self, x, d):
        """Compute step length to boundary from x in direction d.

        It computes the max step length from x to the boundary of the
        nonnegative orthant in the direction d. Also return the component index
        responsible for cutting the steplength the most (or -1 if no such index
        exists).
        """
        self.log.debug('Computing step length to boundary')
        whereneg = np.where(d < 0)[0]
        if len(whereneg) > 0:
            dxneg = -x[whereneg] / d[whereneg]
            kmin = np.argmin(dxneg)
            stepmax = min(1.0, dxneg[kmin])
            if stepmax == 1.0:
                kmin = -1
            else:
                kmin = whereneg[kmin]
        else:
            stepmax = 1.0
            kmin = -1
        return (stepmax, kmin)

    def set_system_matrix(self, initial_guess=False):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        nl = self.nl
        nu = self.nu

        if initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            self.H.put(self.diagQ + self.primal_reg_min**0.5, range(n))
            self.H.put(-self.dual_reg_min**0.5, range(n,n+m))

            self.H.put(-1.0, range(n+m, n+m+nl+nu))
            self.H.put(1.0, range(n+m, n+m+nl), self.all_lb)
            self.H.put(1.0, range(n+m+nl, n+m+nl+nu), self.all_ub)
        else:
            self.log.debug('Setting up matrix for current iteration')
            Lvar = self.qp.Lvar
            Uvar = self.qp.Uvar
            x = self.x
            y = self.y
            zL = self.zL
            zU = self.zU

            self.H.put(self.diagQ + self.primal_reg, range(n))
            self.H.put(-self.dual_reg, range(n,n+m))

            self.H.put(Lvar[self.all_lb] - x[self.all_lb], range(n+m, n+m+nl))
            self.H.put(x[self.all_ub] - Uvar[self.all_ub], range(n+m+nl, n+m+nl+nu))
            self.H.put(zL**0.5, range(n+m, n+m+nl), self.all_lb)
            self.H.put(zU**0.5, range(n+m+nl, n+m+nl+nu), self.all_ub)

        return

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""
        self.log.debug('Updating matrix')
        self.H.put(self.diagQ + self.primal_reg, range(n))
        self.H.put(-self.dual_reg, range(n,n+m))
        return

    def solve_system(self, rhs):
        """Solve the augmented system with right-hand side `rhs`.

        The solution may be iteratively refined based on solver options
        self.itref_threshold and self.nitref.

        The self.LBL object contains all of the solution data.
        """
        self.log.debug('Solving linear system')
        self.LBL.solve(rhs)
        self.LBL.refine(rhs, tol=self.itref_threshold, nitref=self.nitref)

        # Estimate matrix l2-norm condition number.
        if self.estimate_cond:
            rhsNorm = norm2(rhs)
            solnNorm = norm2(self.LBL.x)
            Hop = PysparseLinearOperator(self.H, symmetric=True)
            normH, _ = normest(Hop, tol=1.0e-3)
            if rhsNorm > 0 and solnNorm > 0:
                self.cond_est = solnNorm * normH / rhsNorm
            else:
                self.cond_est = 1.0

        return

    def get_affine_scaling_dxsyz(self, step, x, s, y, z):
        """Split `step` into steps along x, s, y and z.

        his function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        self.log.debug('Recovering affine-scaling step')
        m, n = self.A.shape
        on = self.qp.original_n
        dx = step[:n]
        ds = dx[on:]
        dy = step[n:]
        dz = -z * (1 + ds / s)
        return (dx, ds, dy, dz)

    def update_corrector_rhs(self, rhs, s, z, comp):
        """Update right-hand side for corrector step."""
        self.log.debug('Updating right-hand side for corrector step')
        m, n = self.A.shape
        on = self.qp.original_n
        rhs[on:n] += comp / s - z
        return

    def update_long_step_rhs(self, rhs, pFeas, dFeas, comp, s):
        """Update right-hand side when using long step."""
        self.log.debug('Updating right-hand side for long step')
        m, n = self.A.shape
        on = self.qp.original_n
        rhs[:n] = -dFeas
        rhs[on:n] += comp / s
        rhs[n:] = -pFeas
        return


class RegQPInteriorPointSolver3x3(RegQPInteriorPointSolver):
    """A 3x3 block variant of the regularized interior-point method.

    Linear system is based on the 3x3 block system instead of the reduced 2x2
    block system.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale: Perform row and column equilibration of the constraint
                    matrix [A1 A2] prior to solution (default: `True`).

            :primal_reg: Initial value of primal regularization parameter
                    (default: `1.0`).

            :dual_reg: Initial value of dual regularization parameter
                    (default: `1.0`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.
        """
        super(RegQPInteriorPointSolver3x3, self).__init__(*args, **kwargs)

    def initialize_kkt_matrix(self):
        u"""Create and initialize KKT matrix.

        [ -(Q+ρI)   0        A1'     0   ]
        [  0       -ρI       A2'  Z^{1/2}]
        [  A1       A2       δI      0   ]
        [  0        Z^{1/2}  0       S   ]
        """
        m, n = self.A.shape
        on = self.qp.original_n
        H = PysparseMatrix(size=2 * n + m - on,
                           sizeHint=4 * on + m + self.A.nnz + self.Q.nnz,
                           symmetric=True)

        # The (1,1) block will always be Q (save for its diagonal).
        H[:on, :on] = -self.Q

        # The (2,1) block will always be A. We store it now once and for all.
        H[n:n + m, :n] = self.A
        return H

    def set_initial_guess_system(self):
        """Set linear system for initial guess."""
        m, n = self.A.shape
        on = self.qp.original_n
        self.H.put(-self.diagQ - 1.0e-4, range(on))
        self.H.put(-1.0, range(on, n))
        self.H.put(1.0e-4, range(n, n + m))
        self.H.put(1.0, range(n + m, 2 * n + m - on))
        self.H.put(1.0, range(n + m, 2 * n + m - on), range(on, n))
        return

    def set_initial_guess_rhs(self):
        """Set right-hand side for initial guess."""
        m, n = self.A.shape
        rhs = self.initialize_rhs()
        rhs[n:n + m] = self.b
        return rhs

    def update_initial_guess_rhs(self, rhs):
        """Update right-hand side for initial guess."""
        _, n = self.A.shape
        on = self.qp.original_n
        rhs[:on] = self.c
        rhs[on:] = 0.0
        return

    def initialize_rhs(self):
        """Initialize right-hand side with zeros."""
        m, n = self.A.shape
        on = self.qp.original_n
        return np.zeros(2 * n + m - on)

    def update_linear_system(self, s, z, primal_reg, dual_reg, **kwargs):
        """Update linear system."""
        qp = self.qp
        n = qp.n
        m = qp.m
        on = qp.original_n
        diagQ = self.diagQ
        self.H.put(-diagQ - primal_reg, range(on))
        if primal_reg > 0:
            self.H.put(-primal_reg, range(on, n))
        if dual_reg > 0:
            self.H.put(dual_reg, range(n, n + m))
        self.H.put(np.sqrt(z), range(n + m, 2 * n + m - on), range(on, n))
        self.H.put(s, range(n + m, 2 * n + m - on))
        return

    def set_affine_scaling_rhs(self, rhs, pFeas, dFeas, s, z):
        """Set rhs for affine-scaling step."""
        m, n = self.A.shape
        rhs[:n] = -dFeas
        rhs[n:n + m] = -pFeas
        rhs[n + m:] = -s * np.sqrt(z)
        return

    def get_affine_scaling_dxsyz(self, step, x, s, y, z):
        """Split `step` into steps along x, s, y and z.

        his function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        return self.get_dxsyz(step, x, s, y, z, 0)

    def update_corrector_rhs(self, rhs, s, z, comp):
        """Update right-hand side for corrector step."""
        m, n = self.A.shape
        rhs[n + m:] = -comp / np.sqrt(z)
        return

    def update_long_step_rhs(self, rhs, pFeas, dFeas, comp, s):
        """Update right-hand side when using long step."""
        m, n = self.A.shape
        rhs[:n] = -dFeas
        rhs[n:n + m] = -pFeas
        rhs[n + m:] = -comp
        return

    def get_dxsyz(self, step, x, s, y, z, comp):
        """Split `step` into steps along x, s, y and z.

        This function returns *references*, not copies. Only dz is computed
        from `step` without being a subvector of `step`.
        """
        m, n = self.A.shape
        on = self.qp.original_n
        dx = step[:n]
        ds = step[on:n]
        dy = step[n:n + m]
        dz = np.sqrt(z) * step[n + m:]
        return (dx, ds, dy, dz)
