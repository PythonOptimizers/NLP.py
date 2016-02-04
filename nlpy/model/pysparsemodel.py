try:
    from pysparse.sparse import PysparseMatrix as psp
except:
    print "PySparse is not installed!"

from nlpy.model.nlp import NLPModel
from nlpy.model.snlp import SlackModel
from nlpy.model.qnmodel import QuasiNewtonModel
from nlpy.model.amplpy import AmplModel
from pykrylov.linop.linop import PysparseLinearOperator

import numpy as np


class PySparseNLPModel(NLPModel):
    """
    An `NLPModel` where sparse matrices are returned in PySparse format.
    The `NLPModel`'s `jac` and `hess` methods should return that sparse
    Jacobian and Hessian in coordinate format: (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        vals, rows, cols = super(PySparseNLPModel, self).hess(*args, **kwargs)
        H = psp(size=self.nvar, sizeHint=vals.size, symmetric=True)
        H.put(vals, rows, cols)
        return H

    def jac(self, *args, **kwargs):
        vals, rows, cols = super(PySparseNLPModel,
                                 self).jac(*args, **kwargs)
        J = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        J.put(vals, rows, cols)
        return J


class PySparseAmplModel(PySparseNLPModel, AmplModel):
    # MRO: 1. PySparseAmplModel
    #      2. PySparseNLPModel
    #      3. AmplModel
    #      4. NLPModel
    #
    # Here, `jac` and `hess` are inherited directly from PySparseNPLModel.
    #

    def __init__(self, *args, **kwargs):
        super(PySparseAmplModel, self).__init__(*args, **kwargs)

    def A(self, *args, **kwargs):
        """
        Evaluate sparse Jacobian of the linear part of the
        constraints. Useful to obtain constraint matrix
        when problem is a linear programming problem.
        """
        vals, rows, cols = super(PySparseNLPModel,
                                 self).A(*args, **kwargs)
        A = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        A.put(vals, rows, cols)
        return A

    def jop(self, *args, **kwargs):
        return PysparseLinearOperator(self.jac(*args, **kwargs))


class PySparseSlackModel(SlackModel):
    def __init__(self, nlp, keep_variable_bounds=False, **kwargs):
        if not isinstance(nlp, PySparseNLPModel):
            raise TypeError("The model in `nlp` should be a PySparseNLPModel"
                            "or a derived class of it.")
        super(PySparseSlackModel, self).__init__(nlp,
                                                 keep_variable_bounds=keep_variable_bounds)


    def _jac(self, x, lp=False):
        """
        Helper method to assemble the Jacobian matrix of the constraints of the
        transformed problems. See the documentation of :meth:`jac` for more
        information.

        The positional argument `lp` should be set to `True` only if the problem
        is known to be a linear program. In this case, the evaluation of the
        constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.n ; m = self.m ; nlp = self.nlp
        on = self.nlp.n ; om = self.nlp.m

        lowerC = np.array(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = np.array(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = np.array(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = np.array(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = np.array(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = np.array(nlp.rangeB) ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        # Initialize sparse Jacobian
        nnzJ = 2 * self.nlp.nnzj + m + nrangeC + nbnds + nrangeB  # Overestimate
        J = psp(nrow=m, ncol=n, sizeHint=nnzJ)

        # Insert contribution of general constraints.
        if lp:
            J[:om, :on] = self.nlp.A()
        else:
            J[:om, :on] = self.nlp.jac(x[:on])
        J[upperC, :on] *= -1.0
        J[om:om + nrangeC, :on] = J[rangeC, :on]  # upper side of range const.
        J[om:om + nrangeC, :on] *= -1.0

        # Create a few index lists
        rlowerC = np.array(range(nlowerC)) ; rlowerB = np.array(range(nlowerB))
        rupperC = np.array(range(nupperC)) ; rupperB = np.array(range(nupperB))
        rrangeC = np.array(range(nrangeC)) ; rrangeB = np.array(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(-1.0,      lowerC,  on + rlowerC)
        J.put(-1.0,      upperC,  on + nlowerC + rupperC)
        J.put(-1.0,      rangeC,  on + nlowerC + nupperC + rrangeC)
        J.put(-1.0, om + rrangeC, on + nlowerC + nupperC + nrangeC + rrangeC)

        if self.keep_variable_bounds:
            return J

        # Insert contribution of bound constraints on the original problem
        bot  = om+nrangeC ; J.put( 1.0, bot + rlowerB, lowerB)
        bot += nlowerB    ; J.put( 1.0, bot + rrangeB, rangeB)
        bot += nrangeB    ; J.put(-1.0, bot + rupperB, upperB)
        bot += nupperB    ; J.put(-1.0, bot + rrangeB, rangeB)

        # Insert contribution of slacks on the bound constraints
        bot  = om+nrangeC
        J.put(-1.0, bot + rlowerB, on + nSlacks + rlowerB)
        bot += nlowerB
        J.put(-1.0, bot + rrangeB, on + nSlacks + nlowerB + rrangeB)
        bot += nrangeB
        J.put(-1.0, bot + rupperB, on + nSlacks + nlowerB + nrangeB + rupperB)
        bot += nupperB
        J.put(-1.0, bot + rrangeB, on+nSlacks+nlowerB+nrangeB+nupperB+rrangeB)
        return J

    def hess(self, x, z=None, *args, **kwargs):
        """
        Evaluate the Hessian of the Lagrangian.
        """
        nlp = self.nlp
        if isinstance(nlp, QuasiNewtonModel):
            return self.hop(x, z, *args, **kwargs)

        on = nlp.n

        pi = self.convert_multipliers(z)
        H = psp(nrow=self.n, ncol=self.n, symmetric=True, sizeHint=self.nlp.nnzh)
        H[:on, :on] = self.nlp.hess(x[:on], pi, *args, **kwargs)
        return H
