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
                         matrix A prior to solution (default: `none`).

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
        """
        if not isinstance(qp, PySparseSlackModel):
            msg = 'The QP model must be an instance of SlackModel with sparse'
            msg2 = ' Hessian and Jacobian matrices available.'
            raise TypeError(msg+msg2)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'nlp.cqp')
        self.log = logging.getLogger(logger_name)

        # Either none, abs, or mc29
        self.scale_type = kwargs.get('scale_type', 'none')

        self.qp = qp
        print qp        # Let the user know we have started

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

        # Initialize format strings for display
        fmt_hdr = '%-4s  %-9s' + '  %-8s' * 7
        self.header = fmt_hdr % ('Iter', 'qpObj', 'pFeas', 'dFeas', 'Mu',
                                 'rho_s', 'delta_r', 'alpha_p', 'alpha_d')
        self.format = '%-4d  %-9.2e' + '  %-8.2e' * 7
        self.format0 = '%-4d  %-9.2e' + '  %-8.2e' * 5 + '  %-8s' * 2

        # Additional options to collect
        self.estimate_cond = kwargs.get('estimate_cond', False)
        self.itermax = kwargs.get('itermax', max(100, 10*qp.n))
        self.stoptol = kwargs.get('stoptol', 1.0e-6)
        self.mehrotra_pc = kwargs.get('mehrotra_pc', True)

        # Values to set in case of a problem failure
        self.status = 'fail'
        self.iter = 0
        self.tsolve = 0.0
        self.kktRes = 0.0
        self.qpObj = 0.0

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

            row_max = np.zeros(m)
            col_max = np.zeros(n)

            # Find maximum row values
            for k in range(len(values)):
                row = irow[k]
                val = abs(values[k])
                row_max[row] = max(row_max[row], val)
            row_max[row_max == 0.0] = 1.0

            log.debug('Max row scaling factor = %8.2e' % np.max(row_max))

            # Modified A values after row scaling
            temp_values = values / row_max[irow]

            # Find maximum column values
            for k in range(len(temp_values)):
                col = jcol[k]
                val = abs(temp_values[k])
                col_max[col] = max(col_max[col], val)
            col_max[col_max == 0.0] = 1.0

            log.debug('Max column scaling factor = %8.2e' % np.max(col_max))

            # Invert the computed maximum values to obtain scaling factors
            # By convention, we multiply the matrices by these scaling factors
            self.row_scale = 1./row_max
            self.col_scale = 1./col_max

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

        # Apply the scaling factors to A, b, Q, and c, if available
        if self.row_scale is not None and self.col_scale is not None:
            values *= self.row_scale[irow]
            values *= self.col_scale[jcol]
            self.A.put(values, irow, jcol)

            (values, irow, jcol) = self.Q.find()
            values *= self.col_scale[irow]
            values *= self.col_scale[jcol]
            self.Q.put(values, irow, jcol)

            self.b *= self.row_scale
            self.c *= self.col_scale

            # Recompute the norms to account for the problem scaling
            self.normb = norm2(self.b)
            self.normc = norm2(self.c)
            self.normA = self.A.matrix.norm('fro')
            self.normQ = self.Q.matrix.norm('fro')
            self.diagQ = self.Q.take(range(self.n))

        return

    def solve(self, **kwargs):
        """Solve the problem.

        :returns:

            :x:            final iterate
            :y:            final value of the Lagrange multipliers associated
                           to `Ax = b`
            :zL:           final value of lower-bound multipliers
            :zU:           final value of upper-bound multipliers
            :iter:         total number of iterations
            :kktRes:       final relative residual
            :tsolve:       time to solve the QP
            :long_status:  string describing the exit status.
            :status:       short version of status, used for printing.

        """
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar
        nl = self.nl
        nu = self.nu

        # Setup the problem
        self.scale()
        self.initialize_system()

        # Obtain initial point from Mehrotra's heuristic.
        (self.x, self.y, self.zL, self.zU) = self.set_initial_guess()
        x = self.x
        y = self.y
        zL = self.zL
        zU = self.zU

        # Calculate optimality conditions at initial point
        kktRes = self.check_optimality()
        exitOpt = kktRes <= self.stoptol

        # Set up other stopping conditions
        exitInfeasP = False
        exitInfeasD = False
        exitIter = False
        iter = 0

        # Compute initial perturbation vectors
        # Note: r is primal feas. perturbation, s is dual feas. perturbation
        s = -self.dFeas / self.primal_reg
        r = -self.pFeas / self.dual_reg
        pr_infeas_count = 0
        du_infeas_count = 0
        rho_s = norm2(self.dFeas) / (1 + self.normc)
        rho_s_min = rho_s
        delta_r = norm2(self.pFeas) / (1 + self.normb)
        delta_r_min = delta_r
        mu0 = self.mu

        # Display header and initial point info
        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        output_line = self.format0 % (iter, self.qpObj, self.pResid, self.dResid,
                                     self.dual_gap, rho_s, delta_r, ' ', ' ')
        self.log.info(output_line)

        setup_time = cputime()

        # Main loop.
        while not (exitOpt or exitInfeasP or exitInfeasD or exitIter):

            iter += 1

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

            # Exit immediately if regularization is unsuccessful.
            if not self.LBL.isFullRank:
                self.log.debug('Unable to regularize sufficiently. Exiting')
                break

            # Compute the right-hand side based on the step computation method
            if self.mehrotra_pc:
                # Compute affine-scaling step, i.e. with centering = 0.
                self.set_system_rhs(sigma=0.0)
                self.solve_system(self.rhs)
                dx_aff, dy_aff, dzL_aff, dzU_aff = self.extract_xyz(sigma=0.0)

                # Compute largest allowed primal and dual stepsizes.
                (alpha_p, index_p, is_up_p) = self.max_primal_step_length(dx_aff)
                (alpha_d, index_d, is_up_d) = self.max_dual_step_length(dzL_aff, dzU_aff)

                # Estimate duality gap after affine-scaling step.
                lComp_aff = (zL + alpha_d*dzL_aff)*(x[self.all_lb] + \
                    alpha_p*dx_aff[self.all_lb] - Lvar[self.all_lb])
                uComp_aff = (zU + alpha_d*dzU_aff)*(Uvar[self.all_ub] - \
                    x[self.all_ub] - alpha_p*dx_aff[self.all_ub])

                # Incorporate predictor information for corrector step.
                if (nl + nu) > 0:
                    mu_aff = (lComp_aff.sum() + uComp_aff.sum()) / (nl + nu)
                    sigma = (mu_aff / self.mu)**3
                else:
                    mu_aff = 0.0
                    sigma = 0.0

            else:
                # Use long-step method: Compute centering parameter.
                sigma = min(0.1, 100 * self.mu)

            # Solve augmented system.
            self.set_system_rhs(sigma=sigma)
            self.solve_system(self.rhs)
            dx, dy, dzL, dzU = self.extract_xyz(sigma=sigma)

            # Update regularization parameters before calculating the 
            # step sizes
            self.dual_reg = max(self.dual_reg / 10, self.dual_reg_min)
            self.primal_reg = max(self.primal_reg / 10, self.primal_reg_min)

            # Compute largest allowed primal and dual stepsizes.
            (alpha_p, index_p, is_up_p) = self.max_primal_step_length(dx)
            (alpha_d, index_d, is_up_d) = self.max_dual_step_length(dzL, dzU)

            # Define fraction-to-the-boundary factor and compute the true
            # step sizes
            tau = max(.995, 1.0 - self.mu)

            if self.mehrotra_pc:
                # Compute actual stepsize using Mehrotra's heuristic.

                if index_p == index_d and is_up_p == is_up_d:
                    # If both are -1, do nothing, since the step remains
                    # strictly feasible and alpha_p = alpha_d = 1
                    if index_p != -1:
                        # There is a division by zero in Mehrotra's heuristic
                        # Fall back on classical rule.
                        alpha_p *= tau
                        alpha_d *= tau
                else:
                    mult = 0.01
                    lComp_temp = (zL + alpha_d*dzL)*(x[self.all_lb] + \
                        alpha_p*dx[self.all_lb] - Lvar[self.all_lb])
                    uComp_temp = (zU + alpha_d*dzU)*(Uvar[self.all_ub] - \
                        x[self.all_ub] - alpha_p*dx[self.all_ub])
                    mu_temp = (lComp_temp.sum() + uComp_temp.sum()) / (nl + nu)

                    # If alpha_p < 1.0, compute a gamma_p such that the
                    # complementarity of the updated (x,z) pair is mult*mu_temp
                    if index_p != -1:
                        if is_up_p:
                            ref_index = self.all_ub.index(index_p)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zU[ref_index] + alpha_d*dzU[ref_index])
                            gamma_p -= (Uvar[index_p] - x[index_p])
                            gamma_p /= -(alpha_p*dx[index_p])
                        else:
                            ref_index = self.all_lb.index(index_p)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zL[ref_index] + alpha_d*dzL[ref_index])
                            gamma_p -= (x[index_p] - Lvar[index_p])
                            gamma_p /= (alpha_p*dx[index_p])

                        # If mu_temp is very small, gamma_p = 1. is possible due to
                        # a cancellation error in the gamma_p calculation above.
                        # Therefore, set a maximum value of alpha_p < 1 to prevent
                        # division-by-zero errors later in the program.
                        alpha_p *= min(max(1 - mult, gamma_p), 1. - 1.e-16)

                    # If alpha_d < 1.0, compute a gamma_d such that the
                    # complementarity of the updated (x,z) pair is mult*mu_temp
                    if index_d != -1:
                        if is_up_d:
                            ref_index = self.all_ub.index(index_d)
                            gamma_d = mult * mu_temp
                            gamma_d /= (Uvar[index_d] - x[index_d] - alpha_p*dx[index_d])
                            gamma_d -= zU[ref_index]
                            gamma_d /= (alpha_d*dzU[ref_index])
                        else:
                            ref_index = self.all_lb.index(index_d)
                            gamma_d = mult * mu_temp
                            gamma_d /= (x[index_d] + alpha_p*dx[index_d] - Lvar[index_d])
                            gamma_d -= zL[ref_index]
                            gamma_d /= (alpha_d*dzL[ref_index])

                        # If mu_temp is very small, gamma_d = 1. is possible due to
                        # a cancellation error in the gamma_d calculation above.
                        # Therefore, set a maximum value of alpha_d < 1 to prevent
                        # division-by-zero errors later in the program.
                        alpha_d *= min(max(1 - mult, gamma_d), 1. - 1.e-16)

            else:
                # Use the standard fraction-to-the-boundary rule
                alpha_p *= tau
                alpha_d *= tau

            # Update iterates and perturbation vectors.
            x += alpha_p * dx
            y += alpha_d * dy
            zL += alpha_d * dzL
            zU += alpha_d * dzU
            s = (1 - alpha_p)*s + alpha_p*dx
            r = (1 - alpha_d)*r + alpha_d*dy

            sNorm = norm2(s)
            rNorm = norm2(r)
            rho_s = self.primal_reg * sNorm / (1 + self.normc)
            rho_s_min = min(rho_s_min, rho_s)
            delta_r = self.dual_reg * rNorm / (1 + self.normb)
            delta_r_min = min(delta_r_min, delta_r)

            # Check for dual infeasibility
            if self.mu < 0.01 * self.stoptol * mu0 and \
                    rho_s > rho_s_min * (1.e-6 / self.stoptol) :
                du_infeas_count += 1
                if du_infeas_count > 6:
                    exitInfeasD = True
            else:
                du_infeas_count = 0

            # Check for primal infeasibility
            if self.mu < 0.01 * self.stoptol * mu0 and \
                    delta_r > delta_r_min * (1.e-6 / self.stoptol) :
                pr_infeas_count += 1
                if pr_infeas_count > 6:
                    exitInfeasP = True
                    # continue
            else:
                pr_infeas_count = 0

            # Check for optimality at new point
            kktRes = self.check_optimality()
            exitOpt = kktRes <= self.stoptol

            # Check iteration limit
            exitIter = iter == self.itermax

            # Log updated point info
            if iter % 20 == 0:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))
            output_line = self.format % (iter, self.qpObj, self.pResid, self.dResid,
                                         self.dual_gap, rho_s, delta_r,
                                         alpha_p, alpha_d)
            self.log.info(output_line)

        # Determine solution time
        tsolve = cputime() - setup_time

        self.log.info('-' * len(self.header))

        # Resolve why the iteration stopped and print status
        if exitOpt:
            long_status = 'Optimal solution found'
            status = 'opt'
        elif exitInfeasD:
            long_status = 'Problem seems to be (locally) dual infeasible'
            status = 'dInf'
        elif exitInfeasP:
            long_status = 'Problem seems to be (locally) primal infeasible'
            status = 'pInf'
        elif exitIter:
            long_status = 'Maximum number of iterations reached'
            status = 'iter'
        else:
            long_status = 'Problem could not be regularized sufficiently.'
            status = 'degn'

        self.log.info(long_status)

        # Transfer final values to class members.
        self.iter = iter
        self.kktRes = kktRes
        self.tsolve = tsolve
        self.long_status = long_status
        self.status = status

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
        n = self.n
        m = self.m
        nl = self.nl
        nu = self.nu
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        self.log.debug('Computing initial guess')

        # Let the class know we are initializing the problem for now
        self.initial_guess = True

        # Set up augmented system matrix
        self.set_system_matrix()

        # Analyze and factorize the matrix
        self.LBL = LBLContext(self.H,
            sqd=(self.primal_reg_min > 0.0 and self.dual_reg_min > 0.0))

        # Assemble first right-hand side
        self.set_system_rhs()

        # Solve system and collect solution
        self.solve_system(self.rhs)
        x, _, _, _ = self.extract_xyz()

        # Assemble second right-hand side
        self.set_system_rhs(dual=True)

        # Solve system and collect solution
        self.solve_system(self.rhs)
        _, y, zL_guess, zU_guess = self.extract_xyz()

        # Use Mehrotra's heuristic to compute a strictly feasible starting
        # point for all x and z
        if nl > 0:
            rL_guess = x[self.all_lb] - Lvar[self.all_lb]
            drL = 1.5 + max(0.0, -1.5*np.min(rL_guess))
            dzL = 1.5 + max(0.0, -1.5*np.min(zL_guess))

            rL_shift = drL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
                ((zL_guess + dzL).sum())
            zL_shift = dzL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
                ((rL_guess + drL).sum())

            rL = rL_guess + rL_shift
            zL = zL_guess + zL_shift
            x[self.all_lb] = Lvar[self.all_lb] + rL
        else:
            zL = zL_guess

        if nu > 0:
            rU_guess = Uvar[self.all_ub] - x[self.all_ub]

            drU = 1.5 + max(0.0, -1.5*np.min(rU_guess))
            dzU = 1.5 + max(0.0, -1.5*np.min(zU_guess))

            rU_shift = drU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
                ((zU_guess + dzU).sum())
            zU_shift = dzU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
                ((rU_guess + drU).sum())

            rU = rU_guess + rU_shift
            zU = zU_guess + zU_shift
            x[self.all_ub] = Uvar[self.all_ub] - rU
        else:
            zU = zU_guess

        # An additional normalization step for the range-bounded variables
        #
        # This normalization prevents the shift computed in rL and rU from
        # taking us outside the feasible range, and yields the same final
        # x value whether we take (Lvar + rL*norm) or (Uvar - rU*norm) as x
        if nl > 0 and nu > 0:
            intervals = Uvar[self.qp.rangeB] - Lvar[self.qp.rangeB]
            norm_factors = intervals / (intervals + rL_shift + rU_shift)
            x[self.qp.rangeB] = Lvar[self.qp.rangeB] + rL[self.range_in_lb]*norm_factors

        # Initialization complete
        self.initial_guess = False

        # Check strict feasibility
        if not np.all(x > Lvar) or not np.all(x < Uvar) or \
        not np.all(zL > 0) or not np.all(zU > 0):
            raise ValueError('Initial point not strictly feasible')

        return (x, y, zL, zU)

    def max_primal_step_length(self, dx):
        """Compute the maximum step to the boundary in the primal variables.

        The function also returns the component index that produces the
        minimum steplength. (If the minimum steplength is 1, this value is
        set to -1.)
        """
        self.log.debug('Computing primal step length')
        xl = self.x[self.all_lb]
        xu = self.x[self.all_ub]
        dxl = dx[self.all_lb]
        dxu = dx[self.all_ub]
        l = self.qp.Lvar[self.all_lb]
        u = self.qp.Uvar[self.all_ub]
        eps = 1.e-20

        if self.nl == 0:
            alphaL_max = 1.0
        else:
            # If dxl == 0., shift it slightly to prevent division by zero
            dxl_mod = np.where(dxl == 0., eps, dxl)
            alphaL = np.where(dxl < 0, -(xl - l)/dxl_mod, 1.)
            alphaL_max = min(1.0, alphaL.min())

        if self.nu == 0:
            alphaU_max = 1.0
        else:
            # If dxu == 0., shift it slightly to prevent division by zero
            dxu_mod = np.where(dxu == 0., -eps, dxu)
            alphaU = np.where(dxu > 0, (u - xu)/dxu_mod, 1.)
            alphaU_max = min(1.0, alphaU.min())

        if min(alphaL_max,alphaU_max) == 1.0:
            return (1.0, -1, False)

        if alphaL_max < alphaU_max:
            alpha_max = alphaL_max
            ind_max = self.all_lb[np.argmin(alphaL)]
            is_upper = False
        else:
            alpha_max = alphaU_max
            ind_max = self.all_ub[np.argmin(alphaU)]
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def max_dual_step_length(self, dzL, dzU):
        """Compute the maximum step to the boundary in the dual variables."""
        self.log.debug('Computing dual step length')

        if self.nl == 0:
            alphaL_max = 1.0
        else:
            alphaL = np.where(dzL < 0, -self.zL/dzL, 1.)
            alphaL_max = min(1.0,alphaL.min())

        if self.nu == 0:
            alphaU_max = 1.0
        else:
            alphaU = np.where(dzU < 0, -self.zU/dzU, 1.)
            alphaU_max = min(1.0,alphaU.min())

        if min(alphaL_max,alphaU_max) == 1.0:
            return (1.0, -1, False)

        if alphaL_max < alphaU_max:
            alpha_max = alphaL_max
            ind_max = self.all_lb[np.argmin(alphaL)]
            is_upper = False
        else:
            alpha_max = alphaU_max
            ind_max = self.all_ub[np.argmin(alphaU)]
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and Q blocks of the matrix are also put in place since they
        are common to all problems.
        """
        self.sys_size = self.n + self.m + self.nl + self.nu

        self.H = PysparseMatrix(size=self.sys_size,
            sizeHint=self.nl + self.nu + self.A.nnz + self.Q.nnz + self.sys_size,
            symmetric=True)
        self.H[:self.n, :self.n] = self.Q
        self.H[self.n:self.n+self.m, :self.n] = self.A

        self.rhs = np.zeros(self.sys_size)
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        nl = self.nl
        nu = self.nu

        if self.initial_guess:
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
        n = self.n
        m = self.m
        self.log.debug('Updating matrix')
        self.H.put(self.diagQ + self.primal_reg, range(n))
        self.H.put(-self.dual_reg, range(n,n+m))
        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        m = self.m
        nl = self.nl
        self.log.debug('Setting up linear system right-hand side')

        if self.initial_guess:
            self.rhs[n+m:n+m+nl] = self.qp.Lvar[self.all_lb]
            self.rhs[n+m+nl:] = self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[:n] = 0.
                self.rhs[n:n+m] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] = -self.c
                self.rhs[n:n+m] = 0.
        else:
            sigma = kwargs.get('sigma',0.0)
            self.rhs[:n] = -self.dFeas
            self.rhs[n:n+m] = -self.pFeas
            self.rhs[n+m:n+m+nl] = -self.lComp + sigma*self.mu
            self.rhs[n+m:n+m+nl] *= self.zL**-0.5
            self.rhs[n+m+nl:] = self.uComp - sigma*self.mu
            self.rhs[n+m+nl:] *= self.zU**-0.5

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

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m
        nl = self.nl

        x = self.LBL.x[:n].copy()
        y = -self.LBL.x[n:n+m].copy()
        if self.initial_guess:
            zL = -self.LBL.x[n+m:n+m+nl].copy()
            zU = self.LBL.x[n+m+nl:].copy()
        else:
            zL = -(self.zL**0.5)*self.LBL.x[n+m:n+m+nl].copy()
            zU = (self.zU**0.5)*self.LBL.x[n+m+nl:].copy()

        return x,y,zL,zU

    def check_optimality(self):
        """Compute feasibility and complementarity for the current point"""
        x = self.x
        y = self.y
        zL = self.zL
        zU = self.zU
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        # Residual and complementarity vectors
        Qx = self.Q*x
        self.qpObj = self.q + np.dot(self.c,x) + 0.5*np.dot(x,Qx)
        self.pFeas = self.A*x - self.b
        self.dFeas = Qx + self.c - y*self.A
        self.dFeas[self.all_lb] -= zL
        self.dFeas[self.all_ub] += zU
        self.lComp = zL*(x[self.all_lb] - Lvar[self.all_lb])
        self.uComp = zU*(Uvar[self.all_ub] - x[self.all_ub])

        pFeasNorm = norm2(self.pFeas)
        dFeasNorm = norm2(self.dFeas)
        if (self.nl + self.nu) > 0:
            self.mu = (self.lComp.sum() + self.uComp.sum()) / (self.nl + self.nu)
        else:
            self.mu = 0.0

        # Scaled residual norms and duality gap
        self.pResid = pFeasNorm / (1 + self.normb + self.normA + self.normQ)
        self.dResid = dFeasNorm / (1 + self.normc + self.normA + self.normQ)
        self.dual_gap = self.mu / (1 + abs(np.dot(self.c,x)) + self.normA + self.normQ)

        # Overall residual for stopping condition
        return max(self.pResid, self.dResid, self.dual_gap)


class RegQPInteriorPointSolver2x2(RegQPInteriorPointSolver):
    """A 2x2 block variant of the regularized interior-point method.

    Linear system is based on the (reduced) 2x2 block system instead of
    the 3x3 block system.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix A prior to solution (default: `none`).

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
        """
        super(RegQPInteriorPointSolver2x2, self).__init__(*args, **kwargs)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and Q blocks of the matrix are also put in place since they
        are common to all problems.
        """
        self.sys_size = self.n + self.m

        self.H = PysparseMatrix(size=self.sys_size,
            sizeHint=self.A.nnz + self.Q.nnz + self.sys_size,
            symmetric=True)
        self.H[:self.n, :self.n] = self.Q
        self.H[self.n:self.n+self.m, :self.n] = self.A

        self.rhs = np.zeros(self.sys_size)
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            new_diag = self.diagQ + self.primal_reg_min**0.5
            new_diag[self.all_lb] += 1.0
            new_diag[self.all_ub] += 1.0

            self.H.put(new_diag, range(n))
            self.H.put(-self.dual_reg_min**0.5, range(n,n+m))

        else:
            self.log.debug('Setting up matrix for current iteration')

            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            new_diag = self.diagQ + self.primal_reg
            new_diag[self.all_lb] += self.zL / x_minus_l
            new_diag[self.all_ub] += self.zU / u_minus_x

            self.H.put(new_diag, range(n))
            self.H.put(-self.dual_reg_min**0.5, range(n,n+m))

        return

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""
        self.log.debug('Updating matrix')
        x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
        u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

        new_diag = self.diagQ + self.primal_reg
        new_diag[self.all_lb] += self.zL / x_minus_l
        new_diag[self.all_ub] += self.zU / u_minus_x

        n = self.n
        m = self.m
        self.H.put(new_diag, range(n))
        self.H.put(-self.dual_reg_min**0.5, range(n,n+m))
        self.H.put(-self.dual_reg, range(n,n+m))
        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        m = self.m
        self.log.debug('Setting up linear system right-hand side')

        if self.initial_guess:
            self.rhs[:n] = 0.
            self.rhs[self.all_lb] += self.qp.Lvar[self.all_lb]
            self.rhs[self.all_ub] += self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[n:n+m] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] -= self.c
                self.rhs[n:n+m] = 0.
        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            self.rhs[:n] = -self.dFeas
            self.rhs[n:n+m] = -self.pFeas
            self.rhs[self.all_lb] += -self.zL + sigma*self.mu/x_minus_l
            self.rhs[self.all_ub] += self.zU - sigma*self.mu/u_minus_x

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m

        x = self.LBL.x[:n].copy()
        y = -self.LBL.x[n:n+m].copy()
        if self.initial_guess:
            zL = self.qp.Lvar[self.all_lb] - x[self.all_lb]
            zU = x[self.all_ub] - self.qp.Uvar[self.all_ub]
        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            zL = (-self.zL*(x_minus_l + x[self.all_lb]) + sigma*self.mu)
            zL /= x_minus_l
            zU = (-self.zU*(u_minus_x - x[self.all_ub]) + sigma*self.mu)
            zU /= u_minus_x

        return x,y,zL,zU
