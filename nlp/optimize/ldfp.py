"""A limited-memory DFP method for unconstrained minimization.

A symmetric and positive definite approximation of the Hessian matrix is built
and updated at each iteration following the Davidon-Fletcher-Powell formula.
For efficiency, only the recent observed curvature is incorporated into the
approximation, resulting in a *limited-memory* scheme.

The main idea of this method is that the DFP formula is dual to the BFGS
formula. Therefore, by swapping s and y in the (s,y) pairs, the InverseLBFGS
class updates a limited-memory DFP approximation to the Hessian, rather than a
limited-memory BFGS approximation to its inverse.
"""
from pykrylov.linop import InverseLBFGS

__docformat__ = 'restructuredtext'


# Subclass InverseLBFGS to update a LDFP approximation to the Hessian
# (as opposed to a LBFGS approximation to its inverse).
class LDFP(InverseLBFGS):
    """A limited-memory DFP framework for quasi-Newton methods.

    See the documentation of `InverseLBFGS`.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """Instantiate a limited-memory DFP quasi-Newton approximation."""
        super(LDFP, self).__init__(n, npairs, **kwargs)

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}."""
        # Simply swap s and y.
        InverseLBFGS.store(self, new_y, new_s)


class StructuredLDFP(InverseLBFGS):
    """A structured limited - memory DFP framework for quasi - Newton methods.

    A limited - memory DFP framework for quasi - Newton methods that only
    memorizes updates corresponding to certain variables. This is useful
    when approximating the Hessian of a constraint with a sparse Jacobian.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """See the documentation of `InverseLBFGS` for complete information.

        : parameters:
            : n: Number of variables of the problem.

        : keywords:
            : npairs: Number of {s, y} pairs stored(default: 5)
            : scaling: Enable scaling of the 'initial matrix'. Scaling is done
                      as 'method M3' in the LBFGS paper by Zhou and Nocedal.
                      The scaling factor is s'y/y'y(default: False).
            : vars: List of variables participating in the quasi - Newton
                   update. If `None`, all variables participate.
        """
        self.on = n   # Original value of n.
        self.vars = kwargs.get('vars', None)  # None means all variables.
        if self.vars is None:
            nvars = n
        else:
            nvars = len(self.vars)

        # This next initialization will set self.n to nvars.
        # The original value of n was saved in self.on.
        super(StructuredLDFP, self).__init__(nvars, npairs, **kwargs)

    def store(self, new_s, new_y):
        """Store the new pair {new_s, new_y}.

        This method takes "small" vectors as input, i.e., corresponding to the
        variables participating in the quasi - Newton update.
        """
        InverseLBFGS.store(self, new_y, new_s)

    def matvec(self, v):
        """Compute matrix - vector product with structured DFP approximation.

        Take a small vector and return a small vector giving the contribution
        of the Hessian approximation to the matrix - vector product.
        """
        return InverseLBFGS.matvec(self, v)
