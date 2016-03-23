"""A General module that implements various linesearch schemes."""

__docformat__ = 'restructuredtext'


class LineSearch(object):
    """A generic linesearch class.

    Most methods of this class should be overridden by subclassing.
    """

    def __init__(self, **kwargs):
        self._id = 'Generic Linesearch'
        return

    def _test(self, fTrial, slope, t=1.0, **kwargs):
        """Linesearch condition.

        Given a descent direction `d` for function at the
        current iterate `x`, see if the steplength `t` satisfies
        a specific linesearch condition.

        Must be overridden.
        """
        return True  # Must override

    def search(self, func, x, d, slope, f=None, **kwargs):
        """Compute a steplength `t` satisfying a linesearch condition.

        Given a descent direction `d` for function `func` at the current
        iterate `x`, compute a steplength `t` such that `func(x + t * d)`
        satisfies a linesearch condition when compared to `func(x)`.

        :keywords:
            :func: Pointer to a defined function or to a lambda function.
                   For example, in the univariate case:

                        `test(lambda x: x**2, 2.0, -1, 4.0)`
            :x: Current iterate
            :d: Direction along which doing the backtracking linesearch
            :slope: The directional derivative of `func` at `x` in the
                    direction `d`: slope = f'(x;d) < 0
            :bkmax: Maximum number of backtracking steps
            :f: If given, it should be the value of `func` at `x`.
                If not given, it will be evaluated.
            :fTrial: If given, it should be the value of `func` at `x+d`.
                     If not given, it will be evaluated.

        :returns:
            :xTrial: New iterate `x + t * step` which satisfies a linesearch
                     condition.
            :fTrial: Value of `func` at this new point.
            :t: Final step length.
        """
        # Must override
        if slope >= 0.0:
            raise ValueError("Direction must be a descent direction")
            return None
        t = 1.0
        while not self._test(fTrial, slope, t=t, **kwargs):
            pass
        return t


class ArmijoLineSearch(LineSearch):
    """An Armijo linesearch with backtracking."""

    def __init__(self, **kwargs):
        """Implement the simple Armijo test.

        f(x + t * d) <= f(x) + t * beta * f'(x;d)

        where 0 < beta < 1/2 and f'(x;d) is the directional derivative of f in
        the direction d. Note that f'(x;d) < 0 must be true.

        :keywords:

            :beta:      Value of beta (default 1e-4)
            :tfactor:   Amount by which to reduce the steplength
                        during the backtracking (default 0.1).
        """
        super(ArmijoLineSearch, self).__init__(**kwargs)
        self.beta = max(min(kwargs.get('beta', 1.0e-4), 0.5), 1.0e-10)
        self.tfactor = max(min(kwargs.get('tfactor', 0.1), 0.999), 1.0e-3)
        return

    def _test(self, fTrial, slope, t=1.0, **kwargs):
        """Armijo condition.

        Given a descent direction d for function func at the
        current iterate x, see if the steplength t satisfies
        the Armijo linesearch condition.
        """
        return (fTrial <= f + t * self.beta * slope)

    def search(self, func, x, d, slope, bkmax=5,
               f=None, fTrial=None, **kwargs):
        """Compute a steplength `t` satisfying the Armijo condition.

        Given a descent direction `d` for function `func` at the current
        iterate `x`, compute a steplength `t` such that `func(x + t * d)`
        satisfies the Armijo linesearch condition when compared to `func(x)`.

        :keywords:
            :func: Pointer to a defined function or to a lambda function.
                   For example, in the univariate case:

                        `test(lambda x: x**2, 2.0, -1, 4.0)`
            :x: Current iterate
            :d: Direction along which doing the backtracking linesearch
            :slope: The directional derivative of `func` at `x` in the
                    direction `d`: slope = f'(x;d) < 0
            :bkmax: Maximum number of backtracking steps
            :f: If given, it should be the value of `func` at `x`.
                If not given, it will be evaluated.
            :fTrial: If given, it should be the value of `func` at `x+d`.
                     If not given, it will be evaluated.

        :returns:
            :xTrial: New iterate `x + t * step` which satisfies the
                     Armijo condition.
            :fTrial: Value of `func` at this new point.
            :t: Final step length.
        """
        if f is None:
            f = func(x)
        xTrial = x + d

        if fTrial is None:
            fTrial = func(xTrial)

        if slope >= 0.0:
            raise ValueError("Direction must be a descent direction")
            return None
        bk = 0
        t = 1.0
        while not self._test(fTrial, slope, t=t, **kwargs) and bk < bkmax:
            bk += 1
            t *= self.tfactor
            xTrial = x + t * d
            fTrial = func(xTrial)
        return (xTrial, fTrial, t)


if __name__ == '__main__':

    # Simple example:
    #    steepest descent method
    #    with Armijo backtracking
    from numpy import array, dot
    from nlp.tools.norms import norm_infty

    def rosenbrock(x):
        """Usual 2D Rosenbrock function."""
        return 10.0 * (x[1]-x[0]**2)**2 + (1-x[0])**2

    def rosenbrockxy(x, y):
        """For plotting purposes."""
        return rosenbrock((x, y))

    def rosengrad(x):
        """Gradient of `rosenbrock`."""
        return array([-40.0 * (x[1] - x[0]**2) * x[0] - 2.0 * (1-x[0]),
                      20.0 * (x[1] - x[0]**2)], 'd')

    ALS = ArmijoLineSearch(tfactor=0.2)
    x = array([-0.5, 1.0], 'd')
    xmin = xmax = x[0]
    ymin = ymax = x[1]
    f = rosenbrock(x)
    grad = rosengrad(x)
    d = -grad
    slope = dot(grad, d)
    t = 0.0
    tlist = []
    xlist = [x[0]]
    ylist = [x[1]]
    iter = 0
    print '%-d\t%-g\t%-g\t%-g\t%-g\t%-g\t%-g' % (iter, f, norm_infty(grad),
                                                 x[0], x[1], t, slope)
    while norm_infty(grad) > 1.0e-5:

        iter += 1

        # Perform linesearch
        (x, f, t) = ALS.search(rosenbrock, x, d, slope, ftrial=f)
        tlist.append(t)

        xlist.append(x[0])
        ylist.append(x[1])
        xmin = min(xmin, x[0])
        xmax = max(xmax, x[0])
        ymin = min(ymin, x[1])
        ymax = max(ymax, x[1])
        grad = rosengrad(x)
        d = -grad
        slope = dot(grad, d)
        print '%-d\t%-g\t%-g\t%-g\t%-g\t%-g\t%-g' % (iter, f, norm_infty(grad),
                                                     x[0], x[1], t, slope)

    try:
        from pylab import *
    except:
        import sys
        sys.stderr.write('If you had Matplotlib, you would be looking\n')
        sys.stderr.write('at a contour plot right now...\n')
        sys.exit(0)
    xx = arange(-1.5, 1.5, 0.01)
    yy = arange(-0.5, 1.5, 0.01)
    XX, YY = meshgrid(xx, yy)
    ZZ = rosenbrockxy(XX, YY)
    plot(xlist, ylist, 'r-', lw=1.5)
    plot([xlist[0]], [ylist[0]], 'go', [xlist[-1]], [ylist[-1]], 'go')
    contour(XX, YY, ZZ, 30, linewidths=1.5, alpha=0.75, origin='lower')
    title('Steepest descent with Armijo linesearch on the Rosenbrock function')
    show()
