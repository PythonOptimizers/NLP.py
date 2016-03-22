cimport numpy as cnp

cimport cython

cdef extern from "math.h":
        double sqrt(double x) nogil

@cython.cdivision(True)
cdef tuple dcstep(double stx, double fx, double dx, double sty, double fy, double dy,
                  double stp, double fp, double dp, bint brackt, double stpmin,
                  double stpmax):
    """
     This subroutine computes a safeguarded step for a search
     procedure and updates an interval that contains a step that
     satisfies a sufficient decrease and a curvature condition.

     The parameter stx contains the step with the least function
     value. If brackt is set to True then a minimizer has
     been bracketed in an interval with endpoints stx and sty.
     The parameter stp contains the current step.
     The subroutine assumes that if brackt is set to True then

           min(stx,sty) < stp < max(stx,sty),

     and that the derivative at stx is negative in the direction
     of the step.

     :parameters:
       stx is a double precision variable.
         On entry stx is the best step obtained so far and is an
            endpoint of the interval that contains the minimizer.
         On exit stx is the updated best step.

       fx is a double precision variable.
         On entry fx is the function at stx.
         On exit fx is the function at stx.

       dx is a double precision variable.
         On entry dx is the derivative of the function at
            stx. The derivative must be negative in the direction of
            the step, that is, dx and stp - stx must have opposite
            signs.
         On exit dx is the derivative of the function at stx.

       sty is a double precision variable.
         On entry sty is the second endpoint of the interval that
            contains the minimizer.
         On exit sty is the updated endpoint of the interval that
            contains the minimizer.

       fy is a double precision variable.
         On entry fy is the function at sty.
         On exit fy is the function at sty.

       dy is a double precision variable.
         On entry dy is the derivative of the function at sty.
         On exit dy is the derivative of the function at the exit sty.

       stp is a double precision variable.
         On entry stp is the current step. If brackt is set to True
            then on input stp must be between stx and sty.
         On exit stp is a new trial step.

       fp is a double precision variable.
         On entry fp is the function at stp
         On exit fp is unchanged.

       dp is a double precision variable.
         On entry dp is the the derivative of the function at stp.
         On exit dp is unchanged.

       brackt is an logical variable.
         On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to False
         On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to True

       stpmin is a double precision variable.
         On entry stpmin is a lower bound for the step.
         On exit stpmin is unchanged.

       stpmax is a double precision variable.
         On entry stpmax is an upper bound for the step.
         On exit stpmax is unchanged.

    """
    cdef:
        double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta

    sgnd = dp*(dx/abs(dx))

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.
    if fp > fx:
        theta = 3*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta),abs(dx),abs(dp))
        gamma = s*sqrt((theta/s)**2-(dx/s)*(dp/s))
        if stp < stx:
            gamma = -gamma
        p = (gamma-dx) + theta
        q = ((gamma-dx)+gamma) + dp
        r = p/q
        stpc = stx + r*(stp-stx)
        stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2)*(stp-stx)
        if abs(stpc-stx) < abs(stpq-stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq-stpc)/2
        brackt = True

    # Second case: A lower function value and derivatives of opposite
    # sign. The minimum is bracketed. If the cubic step is farther from
    # stp than the secant step, the cubic step is taken, otherwise the
    # secant step is taken.
    elif sgnd < 0:
        theta = 3*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta),abs(dx),abs(dp))
        gamma = s*sqrt((theta/s)**2-(dx/s)*(dp/s))
        if stp > stx:
            gamma = -gamma
        p = (gamma-dp) + theta
        q = ((gamma-dp)+gamma) + dx
        r = p/q
        stpc = stp + r*(stx-stp)
        stpq = stp + (dp/(dp-dx))*(stx-stp)
        if abs(stpc-stp) > abs(stpq-stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True

    # Third case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative decreases.
    elif abs(dp) < abs(dx):

        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = 3*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta),abs(dx),abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s*sqrt(max(0,(theta/s)**2-(dx/s)*(dp/s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma-dp) + theta
        q = (gamma+(dx-dp)) + gamma
        r = p/q
        if r < 0 and gamma != 0:
            stpc = stp + r*(stx-stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin

        stpq = stp + (dp/(dp-dx))*(stx-stp)

        if brackt:
            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc-stp) < abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp+0.66*(sty-stp),stpf)
            else:
                stpf = max(stp+0.66*(sty-stp),stpf)

        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if abs(stpc-stp) > abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq

            stpf = min(stpmax,stpf)
            stpf = max(stpmin,stpf)

    # Fourth case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative does not decrease. If the
    # minimum is not bracketed, the step is either stpmin or stpmax,
    # otherwise the cubic step is taken.
    else:
        if brackt:
            theta = 3*(fp-fy)/(sty-stp) + dy + dp
            s = max(abs(theta),abs(dy),abs(dp))
            gamma = s*sqrt((theta/s)**2-(dy/s)*(dp/s))
            if stp > sty:
                gamma = -gamma
            p = (gamma-dp) + theta
            q = ((gamma-dp)+gamma) + dy
            r = p/q
            stpc = stp + r*(sty-stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # Update the interval which contains a minimizer.
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new step.
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, brackt

@cython.boundscheck(False)
def dcsrch(double stp, double f, double g, double ftol, double gtol, double xtol,
           str task, double stpmin, double stpmax,
           cnp.ndarray[cnp.int32_t,ndim=1] isave,
           cnp.ndarray[cnp.float64_t,ndim=1] dsave):
    """
    This subroutine finds a step that satisfies a sufficient
    decrease condition and a curvature condition.

    Each call of the subroutine updates an interval with
    endpoints stx and sty. The interval is initially chosen
    so that it contains a minimizer of the modified function

       psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

    If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    interval is chosen so that it contains a minimizer of f.

    The algorithm is designed to find a step that satisfies
    the sufficient decrease condition

       f(stp) <= f(0) + ftol*stp*f'(0),

    and the curvature condition

       abs(f'(stp)) <= gtol*abs(f'(0)).

    If ftol is less than gtol and if, for example, the function
    is bounded below, then there is always a step which satisfies
    both conditions.

    If no step can be found that satisfies both conditions, then
    the algorithm stops with a warning. In this case stp only
    satisfies the sufficient decrease condition.

    :parameters:
    stp is a double precision variable.
     On entry stp is the current estimate of a satisfactory
        step. On initial entry, a positive initial estimate
        must be provided.
     On exit stp is the current estimate of a satisfactory step
        if task = 'FG'. If task = 'CONV' then stp satisfies
        the sufficient decrease and curvature condition.

    f is a double precision variable.
     On initial entry f is the value of the function at 0.
        On subsequent entries f is the value of the
        function at stp.
     On exit f is the value of the function at stp.

    g is a double precision variable.
     On initial entry g is the derivative of the function at 0.
        On subsequent entries g is the derivative of the
        function at stp.
     On exit g is the derivative of the function at stp.

    ftol is a double precision variable.
     On entry ftol specifies a nonnegative tolerance for the
        sufficient decrease condition.
     On exit ftol is unchanged.

    gtol is a double precision variable.
     On entry gtol specifies a nonnegative tolerance for the
        curvature condition.
     On exit gtol is unchanged.

    xtol is a double precision variable.
     On entry xtol specifies a nonnegative relative tolerance
        for an acceptable step. The subroutine exits with a
        warning if the relative difference between sty and stx
        is less than xtol.
     On exit xtol is unchanged.

    task is a character variable of length at least 60.
     On initial entry task must be set to 'START'.
     On exit task indicates the required action:

        If task(1:2) = 'FG' then evaluate the function and
        derivative at stp and call dcsrch again.

        If task(1:4) = 'CONV' then the search is successful.

        If task(1:4) = 'WARN' then the subroutine is not able
        to satisfy the convergence conditions. The exit value of
        stp contains the best point found during the search.

        If task(1:5) = 'ERROR' then there is an error in the
        input arguments.

     On exit with convergence, a warning or an error, the
        variable task contains additional information.

    stpmin is a double precision variable.
     On entry stpmin is a nonnegative lower bound for the step.
     On exit stpmin is unchanged.

    stpmax is a double precision variable.
     On entry stpmax is a nonnegative upper bound for the step.
     On exit stpmax is unchanged.

    isave is an integer work array of dimension 2.

    dsave is a double precision work array of dimension 13.
    """
    cdef:
        double xtrapl=1.1, xtrapu=4
        bint brackt
        int stage
        double finit, ftest, fm, fx, fxm, fy, fym
        double ginit, gtest, gm, gx, gxm, gy, gym, stx, sty, stmin, stmax
        double width, width1

    # Initialization block.
    if task[0:5] == 'START':

        # Check the input arguments for errors.
        if stp < stpmin: task = 'ERROR: STP < STPMIN'
        if stp > stpmax: task = 'ERROR: STP > STPMAX'
        if g >= 0: task = 'ERROR: INITIAL G >= 0'
        if ftol < 0: task = 'ERROR: FTOL < 0'
        if gtol < 0: task = 'ERROR: GTOL < 0'
        if xtol < 0: task = 'ERROR: XTOL < 0'
        if stpmin < 0: task = 'ERROR: STPMIN < 0'
        if stpmax < stpmin: task = 'ERROR: STPMAX < STPMIN'

        # Exit if there are errors on input.
        if task[0:5] == 'ERROR':
            return stp, task, isave, dsave

        # Initialize local variables.
        brackt = False
        stage = 1
        finit = f
        ginit = g
        gtest = ftol*ginit
        width = stpmax - stpmin
        width1 = width/0.5

        # The variables stx, fx, gx contain the values of the step,
        # function, and derivative at the best step.
        # The variables sty, fy, gy contain the value of the step,
        # function, and derivative at sty.
        # The variables stp, f, g contain the values of the step,
        # function, and derivative at stp.
        stx = 0
        fx = finit
        gx = ginit
        sty = 0
        fy = finit
        gy = ginit
        stmin = 0
        stmax = stp + xtrapu*stp
        task = 'FG'

        # Save local variables.
        save_local_variables(brackt, <int *> cnp.PyArray_DATA(isave), <double *> cnp.PyArray_DATA(dsave), stage, ginit, gtest, gx, gy,
                             finit, fx, fy, stx, sty, stmin, stmax, width, width1)
        return stp, task, isave, dsave
        # return stp, f, g, task, isave, dsave

    else:

        # Restore local variables.
        if isave[0] == 1:
            brackt = True
        else:
            brackt = False
        stage = isave[1]
        ginit = dsave[0]
        gtest = dsave[1]
        gx = dsave[2]
        gy = dsave[3]
        finit = dsave[4]
        fx = dsave[5]
        fy = dsave[6]
        stx = dsave[7]
        sty = dsave[8]
        stmin = dsave[9]
        stmax = dsave[10]
        width = dsave[11]
        width1 = dsave[12]

    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    # algorithm enters the second stage.

    ftest = finit + stp*gtest
    if stage == 1 and f <= ftest and g >= 0: stage = 2

    # Test for warnings.
    if brackt and (stp <= stmin or stp >= stmax):
        task = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
    if brackt and stmax-stmin <= xtol*stmax:
        task = 'WARNING: XTOL TEST SATISFIED'
    if stp == stpmax and f <= ftest and g <= gtest:
        task = 'WARNING: STP = STPMAX'
    if stp == stpmin and (f > ftest or g >= gtest):
        task = 'WARNING: STP = STPMIN'

    # Test for convergence.
    if f <= ftest and abs(g) <= gtol*(-ginit):
        task = 'CONVERGENCE'

    # Test for termination.
    if task[0:4] == 'WARN' or task[0:4] == 'CONV':
        # Save local variables.
        save_local_variables(brackt, <int *> cnp.PyArray_DATA(isave), <double *> cnp.PyArray_DATA(dsave), stage, ginit, gtest, gx, gy,
                             finit, fx, fy, stx, sty, stmin, stmax, width, width1)
        return stp, task, isave, dsave
        # return stp, f, g, task, isave, dsave

    # A modified function is used to predict the step during the
    # first stage if a lower function value has been obtained but
    # the decrease is not sufficient.

    if stage == 1 and f <= fx and f > ftest:

        # Define the modified function and derivative values.
        fm = f - stp*gtest
        fxm = fx - stx*gtest
        fym = fy - sty*gtest
        gm = g - gtest
        gxm = gx - gtest
        gym = gy - gtest

        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fxm, gxm, sty, fym, gym, stp, brackt = dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

        # Reset the function and derivative values for f.
        fx = fxm + stx*gtest
        fy = fym + sty*gtest
        gx = gxm + gtest
        gy = gym + gtest
    else:
        # Call dcstep to update stx, sty, and to compute the new step.
        stx, fx, gx, sty, fy, gy, stp, brackt = dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)

    # Decide if a bisection step is needed.
    if brackt:
        if abs(sty-stx) >= 0.66*width1:
            stp = stx + 0.5*(sty-stx)
        width1 = width
        width = abs(sty-stx)

    # Set the minimum and maximum steps allowed for stp.
    if brackt:
        stmin = min(stx,sty)
        stmax = max(stx,sty)
    else:
        stmin = stp + xtrapl*(stp-stx)
        stmax = stp + xtrapu*(stp-stx)

    # Force the step to be within the bounds stpmax and stpmin.
    stp = max(stp,stpmin)
    stp = min(stp,stpmax)

    # If further progress is not possible, let stp be the best
    # point obtained during the search.
    if brackt and (stp <= stmin or stp >= stmax) or \
       (brackt and stmax-stmin <= xtol*stmax):
        stp = stx

    # Obtain another function and derivative.
    task = 'FG'

    # Save local variables.
    save_local_variables(brackt, <int *> cnp.PyArray_DATA(isave), <double *> cnp.PyArray_DATA(dsave), stage, ginit, gtest, gx, gy,
                         finit, fx, fy, stx, sty, stmin, stmax, width, width1)
    return stp, task, isave, dsave
    # return stp, f, g, task, isave, dsave


@cython.boundscheck(False)
cdef save_local_variables(bint brackt, int * isave,
                          double * dsave, int stage,
                          double ginit, double gtest, double gx, double gy,
                          double finit, double fx, double fy,
                          double stx, double sty, double stmin, double stmax,
                          double width, double width1):
    if brackt:
        isave[0] = 1
    else:
        isave[0] = 0
    isave[1] = stage
    dsave[0] = ginit
    dsave[1] = gtest
    dsave[2] = gx
    dsave[3] = gy
    dsave[4] = finit
    dsave[5] = fx
    dsave[6] = fy
    dsave[7] = stx
    dsave[8] = sty
    dsave[9] = stmin
    dsave[10] = stmax
    dsave[11] = width
    dsave[12] = width1
