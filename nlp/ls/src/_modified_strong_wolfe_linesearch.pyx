import numpy as np
cimport numpy as cnp

cimport cython

cdef extern from "math.h":
        double sqrt(double x) nogil

@cython.cdivision(True)
cdef tuple mcstep(double stx, double fx, double dx, double sty, double fy, double dy,
                  double stp, double fp, double dp, bint brackt,
                  double stpmin, double stpmax, int info):
    """
     The purpose of mcstep is to compute a safeguarded step for a linesearch and
     to update an interval of uncertainty for a minimizer of the function.

     The parameter stx contains the step with the least function value. the
     parameter stp contains the current step. It is assumed that the derivative
     at stx is negative in the direction of the step. If brackt is set true then
     a minimizer has been bracketed in an interval of uncertainty with endpoints
     stx and sty.

     :parameters:
       stx, fx and dx are variables which specify the step,
         the function, and the derivative at the best step obtained
         so far. The derivative must be negative in the direction
         of the step, that is, dx and stp-stx must have opposite
         signs. On output these parameters are updated appropriately.

       sty, fy and dy are variables which specify the step,
         the function, and the derivative at the other endpoint of
         the interval of uncertainty. On output these parameters are
         updated appropriately.

       stp, fp and dp are variables which specify the step,
         the function, and the derivative at the current step.
         if brackt is settruethen on input stp must be
         between stx and sty. On output stp is set to the new step.

       brackt is a logical variable which specifies if a minimizer
         has been bracketed. If the minimizer has not been bracketed
         then on input brackt must be setfalse if the minimizer
         is bracketed then on output brackt is settrue

       stpmin and stpmax are input variables which specify lower
        andupper bounds for the step.

       info is an integer output variable set as follows:
         if info = 1,2,3,4,5, then the step has been computed
         according to one of the five cases below. Otherwise
         info = 0 and this indicates improper input parameters.
    """

    cdef double gamma, p, q, r, s, sgnd, stpC, stpF, stpQ, theta
    cdef bint bound

    info = 0

    # Check the input parameters for errors.
    if ((brackt and (stp <= min(stx,sty) or stp >= max(stx,sty))) \
        or dx*(stp-stx) >= 0. or stpmax < stpmin):
        return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info

    # Determine if the derivatives have opposite sign.
    sgnd = dp*(dx/abs(dx))

    # First case. a higher function value.
    # The minimum is bracketed. If the cubic step is closer
    # to stx than the quadratic step, the cubic step is taken,
    # else the average of the cubic and quadratic steps is taken.
    if fp > fx:
        info = 1
        bound = True
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp < stx: gamma *= -1
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p/q
        stpC = stx + r*(stp - stx)
        stpQ = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2)*(stp - stx)
        if abs(stpC-stx) < abs(stpQ-stx):
            stpF = stpC
        else:
            stpF = stpC + (stpQ - stpC)/2
        brackt = True

    # Second case. A lower function valueandderivatives of
    # opposite sign. The minimum is bracketed. If the cubic
    # step is closer to stx than the quadratic (secant) step,
    # the cubic step is taken, else the quadratic step is taken.
    elif sgnd < 0.0:
        info = 2
        bound = False
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s*sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp > stx: gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p/q
        stpC = stp + r*(stx - stp)
        stpQ = stp + (dp/(dp-dx))*(stx - stp)
        if abs(stpC-stp) > abs(stpQ-stp):
            stpF = stpC
        else:
            stpF = stpQ
        brackt = True

    # Third case. A lower function value, derivatives of the
    # same sign and the magnitude of the derivative decreases.
    # the cubic step is only used if the cubic tends to infinity
    # in the direction of the step or if the minimum of the cubic
    # is beyond stp. otherwise the cubic step is defined to be
    # either stpmin or stpmax. the quadratic (secant) step is also
    # computed and if the minimum is bracketed then the the step
    # closest to stx is taken, else the step farthest away is taken.
    elif abs(dp) < abs(dx):
        info = 3
        bound = True
        theta = 3*(fx - fp)/(stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # The case gamma = 0 only arises if the cubic does not tend
        # to infinity in the direction of the step.
        gamma = s*sqrt(max(0.0, (theta/s)**2 - (dx/s)*(dp/s)))
        if stp > stx: gamma *= -1
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p/q
        if r < 0.0 and gamma != 0.0:
            stpC = stp + r*(stx - stp)
        elif stp > stx:
            stpC = stpmax
        else:
            stpC = stpmin
        stpQ = stp + (dp/(dp-dx))*(stx - stp)
        if brackt:
            if abs(stp-stpC) < abs(stp-stpQ):
               stpF = stpC
            else:
               stpF = stpQ
        else:
            if abs(stp-stpC) > abs(stp-stpQ):
               stpF = stpC
            else:
               stpF = stpQ

    # Fourth case. a lower function value, derivatives of the
    # same sign and the magnitude of the derivative does
    # not decrease. If the minimum is not bracketed, the step
    # is either stpmin or stpmax, else the cubic step is taken.
    else:
        info = 4
        bound = False
        if brackt:
            theta = 3*(fp - fy)/(sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s*sqrt((theta/s)**2 - (dy/s)*(dp/s))
            if stp > sty: gamma *= -1
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p/q
            stpC = stp + r*(sty - stp)
            stpF = stpC
        elif stp > stx:
            stpF = stpmax
        else:
            stpF = stpmin

    # Update the interval of uncertainty. this update does not
    # depend on the new step or the case analysis above.
    if fp > fx:
         sty = stp
         fy = fp
         dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    # Compute the new stepandsafeguard it.
    stpF = min(stpmax, stpF)
    stpF = max(stpmin, stpF)
    stp = stpF
    if brackt and bound:
        if sty > stx:
            stp = min(stx+0.66*(sty-stx), stp)
        else:
            stp = max(stx+0.66*(sty-stx), stp)
    return stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, info


def mcsrch(int n, cnp.ndarray[cnp.float64_t, ndim=1] x, double f,
           cnp.ndarray[cnp.float64_t, ndim=1] g,
           cnp.ndarray[cnp.float64_t, ndim=1] s,
           double stp, double ftol, double gtol, double xtol, double stpmin,
           double stpmax, int maxfev, int info, int nfev,
           cnp.ndarray[cnp.int32_t,ndim=1] isave,
           cnp.ndarray[cnp.float64_t,ndim=1] dsave,
           cnp.ndarray[cnp.float64_t, ndim=1] wa):
    """
    a slight modification of the subroutine csrch of more' and thuente.
    the changes are to allow reverse communication, and do not affect
    the performance of the routine.

    the purpose of mcsrch is to find a step which satisfies
    a sufficient decrease condition and a curvature condition.

    at each stage the subroutine updates an interval of
    uncertainty with endpoints stx and sty. the interval of
    uncertainty is initially chosen so that it contains a
    minimizer of the modified function

      f(x+stp*s) - f(x) - ftol*stp*(gradf(x)'s).

    if a step is obtained for which the modified function
    has a nonpositive function value and nonnegative derivative,
    then the interval of uncertainty is chosen so that it
    contains a minimizer of f(x+stp*s).

    the algorithm is designed to find a step which satisfies
    the sufficient decrease condition

       f(x+stp*s) .le. f(x) + ftol*stp*(gradf(x)'s),

    and the curvature condition

       abs(gradf(x+stp*s)'s)) .le. gtol*abs(gradf(x)'s).

    if ftol is less than gtol and if, for example, the function
    is bounded below, then there is always a step which satisfies
    both conditions. if no step can be found which satisfies both
    conditions, then the algorithm usually stops when rounding
    errors prevent further progress. in this case stp only
    satisfies the sufficient decrease condition.

    :parameters:
        n is a positive integer input variable set to the number
         of variables.

        x is an array of length n. on input it must contain the
         base point for the line search. On output it contains
         x + stp*s.

        f is a variable. on input it must contain the value of f
         at x. on output it contains the value of f at x + stp*s.

        g is an array of length n. On input it must contain the
         gradient of f at x. On output it contains the gradient
         of f at x + stp*s.

        s is an input array of length n which specifies the
         search direction.

        stp is a nonnegative variable. on input stp contains an
         initial estimate of a satisfactory step. On output
         stp contains the final estimate.

        ftol and gtol are nonnegative input variables. (In this reverse
         communication implementation gtol is defined in a common
         statement.) Termination occurs when the sufficient decrease
         condition and the directional derivative condition are
         satisfied.

        xtol is a nonnegative input variable. Termination occurs
         when the relative width of the interval of uncertainty
         is at most xtol.

        stpmin and stpmax are nonnegative input variables which
         specify lower and upper bounds for the step. (In this reverse
         communication implementation they are defined in a common
         statement).

        maxfev is a positive integer input variable. Termination
         occurs when the number of calls to fcn is at least
         maxfev by the end of an iteration.

        info is an integer output variable set as follows:

         info = 0  improper input parameters.

         info =-1  a return is made to compute the function and gradient.

         info = 1  the sufficient decrease condition and the
                   directional derivative condition hold.

         info = 2  relative width of the interval of uncertainty
                   is at most xtol.

         info = 3  number of calls to fcn has reached maxfev.

         info = 4  the step is at the lower bound stpmin.

         info = 5  the step is at the upper bound stpmax.

         info = 6  rounding errors prevent further progress.
                   There may not be a step which satisfies the
                   sufficient decrease and curvature conditions.
                   tolerances may be too small.

        nfev is an integer output variable set to the number of
         calls to fcn.

        wa is a work array of length n.
    """
    cdef:
        int infoc, j
        bint brackt, stage1
        double dg=0, dgm=0, dginit=0, dgtest=0, dgx=0, dgxm=0, dgy=0, dgym=0, finit=0, ftest1=0
        double fm=0, fx=0, fxm=0, fy=0, fym=0, stx=0, sty=0, stmin=0, stmax=0, width=0, width1=0, xtrapf=0

    xtrapf = 4.0

    if info != -1:
        infoc = 1

        # Check the input parameters for errors.
        if n <= 0 or stp <= 0 or ftol < 0 or gtol < 0 or xtol < 0 \
           or stpmin < 0 or stpmax < stpmin or maxfev < 0:
            return x, f, g, stp, info, nfev, wa

        # Compute the initial gradient in the search direction
        # and check that s is a descent direction.
        dginit = np.dot(g, s)
        if dginit >= 0:
            raise ValueError("The search direction is not a descent direction!")

        # Initialize local variables.
        brackt = False
        stage1= True
        nfev = 0
        finit = f
        dgtest = ftol*dginit
        width = stpmax-stpmin
        width1 = width/0.5
        wa = x.copy()

        # The variables stx, fx, dgx contain the values of the step,
        # function, and directional derivative at the best step.
        # the variables sty, fy, dgy contain the value of the step,
        # function, and derivative at the other endpoint of
        # the interval of uncertainty.
        # the variables stp, f, dg contain the values of the step,
        # function, and derivative at the current step.
        stx = 0
        fx = finit
        dgx = dginit
        sty = 0
        fy = finit
        dgy = dginit
    else:
        # Restore local variables.
        if isave[0] == 1:
            brackt = True
        else:
            brackt = False
        stage1 = isave[1]
        dg = dsave[0]
        dgm = dsave[1]
        dginit = dsave[2]
        dgtest = dsave[3]
        dgx = dsave[4]
        dgxm = dsave[5]
        dgy = dsave[6]
        dgym = dsave[7]
        finit = dsave[8]
        ftest1 = dsave[9]
        fm = dsave[10]
        fx = dsave[11]
        fxm = dsave[12]
        fy = dsave[13]
        fym = dsave[14]
        stx = dsave[15]
        sty = dsave[16]
        stmin = dsave[17]
        stmax = dsave[18]
        width = dsave[19]
        width1 = dsave[20]

    # Start of iteration.
    while True:
        if info != -1:
            # Set the minimum and maximum steps to correspond
            # to the present interval of uncertainty.
            if brackt:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stx
                stmax = stp + xtrapf*(stp-stx)

            # Force the step to be within the bounds stpmax and stpmin.
            stp = max(stp, stpmin)
            stp = min(stp, stpmax)

            # If an unusual termination is to occur then let
            # stp be the lowest point obtained so far.
            if brackt and (stp <= stmin or stp >= stmax) \
               or nfev >= maxfev-1 or infoc == 0 \
               or (brackt and stmax-stmin <= xtol*stmax):
                stp = stx

            # Evaluate the function and gradient at stp
            # and compute the directional derivative.
            # we return to main program to obtain f and g.

            x = wa + stp*s

            # for j in xrange(n):
            #     x[j] = wa[j] + stp*s[j]
            info = -1
            save_local_variables(brackt, <int *> cnp.PyArray_DATA(isave),
                                      <double *> cnp.PyArray_DATA(dsave), stage1,
                                      dg, dgm, dginit, dgtest,
                                      dgx, dgxm, dgy, dgy,
                                      finit, ftest1, fm, fx,
                                      fxm, fy, fym,
                                      stx, sty, stmin, stmax,
                                      width, width1)
            return x, f, g, stp, info, nfev, wa
        else:
            info = 0
            nfev += 1
            dg = np.dot(g, s)
            ftest1 = finit + stp*dgtest

            # Test for convergence.
            if (brackt and (stp <= stmin or stp >= stmax)) or infoc == 0:
                info = 6
            if stp == stpmax and f <= ftest1 and dg <= dgtest:
                info = 5
            if stp == stpmin and (f > ftest1 or dg >= dgtest):
                info = 4
            if nfev >= maxfev:
                info = 3
            if brackt and stmax-stmin <= xtol*stmax:
                info = 2
            if f <= ftest1 and abs(dg) <= gtol*(-dginit):
                info = 1

            # Check for termination.
            if info != 0:
                return x, f, g, stp, info, nfev, wa

            # In the first stage we seek a step for which the modified
            # function has a nonpositive value and nonnegative derivative.
            if stage1 and f <= ftest1 and dg >= min(ftol, gtol)*dginit:
                stage1 = False

            # A modified function is used to predict the step only if
            # we have not obtained a step for which the modified
            # function has a nonpositive function value and nonnegative
            # derivative, and if a lower function value has been
            # obtained but the decrease is not sufficient.
            if stage1 and f <= fx and f > ftest1:

                # Define the modified function and derivative values.
                fm = f - stp*dgtest
                fxm = fx - stx*dgtest
                fym = fy - sty*dgtest
                dgm = dg - dgtest
                dgxm = dgx - dgtest
                dgym = dgy - dgtest

                # Call mcstep to update the interval of uncertainty
                # and to compute the new step.
                stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, infoc = mcstep(stx,
                                                                          fxm,
                                                                          dgxm,
                                                                          sty,
                                                                          fym,
                                                                          dgym,
                                                                          stp,
                                                                          fm,
                                                                          dgm,
                                                                          brackt,
                                                                          stmin,
                                                                          stmax, infoc)
                # Reset the function and gradient values for f.
                fx = fxm + stx*dgtest
                fy = fym + sty*dgtest
                dgx = dgxm + dgtest
                dgy = dgym + dgtest
            else:

                # Call mcstep to update the interval of uncertainty
                # and to compute the new step.
                stx, fx, dgx, sty, dy, dgy, stp, f, dg, brackt, infoc = mcstep(stx,
                                                                      fx,
                                                                      dgx,
                                                                      sty,
                                                                      fy,
                                                                      dgy,
                                                                      stp,
                                                                      f,
                                                                      dg,
                                                                      brackt,
                                                                      stmin,
                                                                      stmax, infoc)

            # Force a sufficient decrease in the size of the
            # interval of uncertainty.
            if brackt:
                if abs(sty-stx) >= 0.66*width1:
                    stp = stx + 0.5*(sty - stx)
                    width1 = width
                    width = abs(sty-stx)


@cython.boundscheck(False)
cdef save_local_variables(bint brackt, int * isave,
                          double * dsave, int stage1,
                          double dg, double dgm, double dginit, double dgtest,
                          double dgx, double dgxm, double dgy, double dgym,
                          double finit, double ftest1, double fm, double fx,
                          double fxm, double fy, double fym,
                          double stx, double sty, double stmin, double stmax,
                          double width, double width1):
    if brackt:
        isave[0] = 1
    else:
        isave[0] = 0
    isave[1] = stage1
    dsave[0] = dg
    dsave[1] = dgm
    dsave[2] = dginit
    dsave[3] = dgtest
    dsave[4] = dgx
    dsave[5] = dgxm
    dsave[6] = dgy
    dsave[7] = dgym
    dsave[8] = finit
    dsave[9] = ftest1
    dsave[10] = fm
    dsave[11] = fx
    dsave[12] = fxm
    dsave[13] = fy
    dsave[14] = fym
    dsave[15] = stx
    dsave[16] = sty
    dsave[17] = stmin
    dsave[18] = stmax
    dsave[19] = width
    dsave[20] = width1
