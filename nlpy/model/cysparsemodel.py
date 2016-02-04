try:
    from cysparse.sparse.ll_mat import LLSparseMatrix
    import cysparse.common_types.cysparse_types as types
except:
    print "CySparse is not installed!"

from nlpy.model.nlp import NLPModel
from nlpy.model.snlp import SlackModel
from nlpy.model.qnmodel import QuasiNewtonModel
from nlpy.model.amplpy import AmplModel
from pykrylov.linop import CysparseLinearOperator


class CySparseNLPModel(NLPModel):
    """
    An `NLPModel` where sparse matrices are returned as CySparse matrices.
    The `NLPModel`'s `jac` and `hess` methods should return sparse
    Jacobian and Hessian in coordinate format: (vals, rows, cols).
    """

    def hess(self, *args, **kwargs):
        """
        Note that `rows`, `cols` and `vals` must represent a LOWER triangular
        sparse matrix in the coordinate format (COO).
        """
        vals, rows, cols = super(CySparseNLPModel, self).hess(*args, **kwargs)
        H = LLSparseMatrix(size=self.nvar, size_hint=vals.size,
                           is_symmetric=True, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)
        H.put_triplet(rows, cols, vals)
        return H

    def jac(self, *args, **kwargs):
        vals, rows, cols = super(CySparseNLPModel, self).jac(*args, **kwargs)
        J = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar,
                           size_hint=vals.size, is_symmetric=False,
                           itype=types.INT64_T, dtype=types.FLOAT64_T)
        J.put_triplet(rows, cols, vals)
        return J


class CySparseAmplModel(CySparseNLPModel, AmplModel):
    # MRO: 1. CySparseAmplModel
    #      2. CySparseNLPModel
    #      3. AmplModel
    #      4. NLPModel
    #
    # Here, `jac` and `hess` are inherited directly from CySparseNPLModel.
    #

    def A(self, *args, **kwargs):
        """
        Evaluate sparse Jacobian of the linear part of the
        constraints. Useful to obtain constraint matrix
        when problem is a linear programming problem.
        """
        vals, rows, cols = super(CySparseAmplModel. self).A(*args, **kwargs)
        A = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar,
                           size_hint=vals.size, is_symmetric=False,
                           type=types.INT64_T, dtype=types.FLOAT64_T)
        A.put_triplet(rows, cols, vals)
        return A

    def jop(self, *args, **kwargs):
        return CysparseLinearOperator(self.jac(*args, **kwargs))


class CySparseSlackModel(SlackModel):
    def __init__(self, nlp, keep_variable_bounds=False, **kwargs):
        if not isinstance(nlp, CySparseNLPModel):
            raise TypeError("The model in `nlp` should be a CySparseNLPModel"
                            "or a derived class of it.")
        super(CySparseSlackModel, self).__init__(nlp,
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
        m = self.m ; nlp = self.nlp
        on = self.nlp.n ; om = self.nlp.m

        lowerC = np.array(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = np.array(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = np.array(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = np.array(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = np.array(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = np.array(nlp.rangeB) ; nrangeB = nlp.nrangeB
        nbnds = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        # Initialize sparse Jacobian
        nnzJ = 2 * self.nlp.nnzj + m + nrangeC + nbnds + nrangeB  # Overestimate
        J = LLSparseMatrix(nrow=self.ncon, ncol=self.nvar, size_hint=nnzJ,
                           is_symmetric=False, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)
        # Insert contribution of general constraints.
        if lp:
            J[:om, :on] = self.nlp.A()
        else:
            J[:om, :on] = self.nlp.jac(x[:on])

        # TODO: NOT WORKING, scaling should be done using J *= D
        # where D is a sparse diagonal matrix containing the scaling factors.
        J[upperC, :on] *= -1.0
        J[om:om + nrangeC, :on] = J[rangeC, :on]  # upper side of range const.
        J[om:om + nrangeC, :on] *= -1.0

        # Create a few index lists
        rlowerC = np.array(range(nlowerC)) ; rlowerB = np.array(range(nlowerB))
        rupperC = np.array(range(nupperC)) ; rupperB = np.array(range(nupperB))
        rrangeC = np.array(range(nrangeC)) ; rrangeB = np.array(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(lowerC,  on + rlowerC, -1.0)
        J.put(upperC,  on + nlowerC + rupperC, -1.0)
        J.put(rangeC,  on + nlowerC + nupperC + rrangeC, -1.0)
        J.put(om + rrangeC, on + nlowerC + nupperC + nrangeC + rrangeC, -1.0)

        if self.keep_variable_bounds:
            return J

        # Insert contribution of bound constraints on the original problem
        bot  = om+nrangeC ; J.put(bot + rlowerB, lowerB,  1.0)
        bot += nlowerB    ; J.put(bot + rrangeB, rangeB,  1.0)
        bot += nrangeB    ; J.put(bot + rupperB, upperB, -1.0)
        bot += nupperB    ; J.put(bot + rrangeB, rangeB, -1.0)

        # Insert contribution of slacks on the bound constraints
        bot = om+nrangeC
        J.put(bot + rlowerB, on + nSlacks + rlowerB, -1.0)
        bot += nlowerB
        J.put(bot + rrangeB, on + nSlacks + nlowerB + rrangeB, -1.0)
        bot += nrangeB
        J.put(bot + rupperB, on + nSlacks + nlowerB + nrangeB + rupperB, -1.0)
        bot += nupperB
        J.put(bot + rrangeB, on+nSlacks+nlowerB+nrangeB+nupperB+rrangeB, -1.0)
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
        H = LLSparseMatrix(size=self.nvar, size_hint=self.nlp.nnzh,
                           is_symmetric=True, itype=types.INT64_T,
                           dtype=types.FLOAT64_T)
        H[:on, :on] = self.nlp.hess(x[:on], pi, *args, **kwargs)
        return H
