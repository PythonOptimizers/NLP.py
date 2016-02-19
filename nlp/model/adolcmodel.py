"""Models where derivatives are computed by ADOL-C."""

from nlp.model.nlpmodel import NLPModel
from nlp.model.pysparsemodel import PySparseNLPModel
from nlp.model.scipymodel import SciPyNLPModel
import numpy as np

try:
    import adolc
except:
    print "ADOL-C is not installed"


class BaseAdolcModel(NLPModel):
    """Model with derivatives computed by ADOL-C.

    A class to represent optimization problems in which derivatives
    are computed via algorithmic differentiation through ADOL-C. By
    default, the Jacobian and Hessian are returned in dense format.
    See the documentation of `NLPModel` for further information.
    """

    def __init__(self, n, m, name='Adolc-Generic', **kwargs):
        """Initialize a model with `n` variables and `m` constraints."""
        super(BaseAdolcModel, self).__init__(n, m, name, **kwargs)

        # Trace objective and constraint functions.
        self._obj_trace_id = None
        self._trace_obj(self.x0)

        self._con_trace_id = None
        self._cons_pos_trace_id = None
        if self.m > 0:
            self._trace_con(self.x0)
            self._trace_cons_pos(self.x0)

        self._lag_trace_id = None
        self._trace_lag(self.x0, self.pi0)

    def _get_trace_id(self):
        """Return an available trace id."""
        return 10 * self._id

    @property
    def obj_trace_id(self):
        """Return the trace id for the objective function."""
        return self._obj_trace_id

    @property
    def con_trace_id(self):
        """Return the trace id for the constraints."""
        return self._con_trace_id

    @property
    def cons_pos_trace_id(self):
        """Return the trace id for the reformulated constraints."""
        return self._cons_pos_trace_id

    @property
    def lag_trace_id(self):
        """Return the trace id for the Lagrangian."""
        return self._lag_trace_id

    def _trace_obj(self, x):
        if self._obj_trace_id is None:
            self._obj_trace_id = self._get_trace_id()
            adolc.trace_on(self._obj_trace_id)
            x = adolc.adouble(x)
            adolc.independent(x)
            y = self.obj(x)
            adolc.dependent(y)
            adolc.trace_off()

    def _trace_con(self, x):
        if self._con_trace_id is None and self.m > 0:
            self._con_trace_id = self._get_trace_id() + 1
            adolc.trace_on(self._con_trace_id)
            x = adolc.adouble(x)
            adolc.independent(x)
            y = self.cons(x)
            adolc.dependent(y)
            adolc.trace_off()

    def _trace_cons_pos(self, x):
        if self._con_trace_id is None:
            self._trace_con(x)
        if self._cons_pos_trace_id is None and self.m > 0:
            self._cons_pos_trace_id = self._get_trace_id() + 2
            adolc.trace_on(self._cons_pos_trace_id)
            x = adolc.adouble(x)
            adolc.independent(x)
            y = self.cons_pos(x)
            adolc.dependent(y)
            adolc.trace_off()

    def _trace_lag(self, x, z):
        self._trace_obj(x)
        self._trace_con(x)
        self._trace_cons_pos(x)
        unconstrained = self.m == 0 and self.nbounds == 0
        if self._lag_trace_id is None:
            if unconstrained:
                self._lag_trace_id = self._obj_trace_id
                return

            self._lag_trace_id = self._get_trace_id() + 3
            adolc.trace_on(self._lag_trace_id)
            x = adolc.adouble(x)
            z = adolc.adouble(z)
            adolc.independent(x)
            adolc.independent(z)
            l = self.lag(x, z)
            adolc.dependent(l)
            adolc.trace_off()

    def _adolc_obj(self, x):
        """Evaluate the objective function."""
        return adolc.function(self._obj_trace_id, x)

    def grad(self, x, **kwargs):
        """Evaluate the objective gradient at x."""
        return self._adolc_grad(x, **kwargs)

    def _adolc_grad(self, x, **kwargs):
        """Evaluate the objective gradient."""
        return adolc.gradient(self._obj_trace_id, x)

    def hess(self, x, z, **kwargs):
        """Return the dense Hessian of the objective at x."""
        xz = np.concatenate((x, z))
        H = adolc.hessian(self._lag_trace_id, xz)
        return H[:self.nvar, :self.nvar]

    def hprod(self, x, z, v, **kwargs):
        """Return the Hessian-vector product at x with v."""
        xz = np.concatenate((x, z))
        v0 = np.concatenate((v, np.zeros(self.ncon)))
        return adolc.hess_vec(self._lag_trace_id, xz, v0)[:self.nvar]

    def _adolc_cons(self, x, **kwargs):
        """Evaluate the constraints from the ADOL-C tape."""
        return adolc.function(self._con_trace_id, x)

    def jac(self, x, **kwargs):
        """Return dense constraints Jacobian at x."""
        return adolc.jacobian(self._con_trace_id, x)

    def jac_pos(self, x, **kwargs):
        """Return dense Jacobian of reformulated constraints at x."""
        return adolc.jacobian(self._cons_pos_trace_id, x)

    def jprod(self, x, v, **kwargs):
        """Return the product of v with the Jacobian at x."""
        return adolc.jac_vec(self._con_trace_id, x, v)

    def jtprod(self, x, v, **kwargs):
        """Return the product of v with the transpose Jacobian at x."""
        return adolc.vec_jac(self._con_trace_id, x, v)


class SparseAdolcModel(BaseAdolcModel):
    """`AdolcModel` with sparse Jacobian and Hessian.

    AdolC must have been built with Colpack support for the sparse
    option to be available.
    """

    def __init__(self, *args, **kwargs):
        """See `BaseAdolcModel.__init__`."""
        super(SparseAdolcModel, self).__init__(*args, **kwargs)
        self.__first_sparse_hess_eval = True
        self.__first_sparse_jac_eval = True

    def hess(self, x, z, **kwargs):
        """Return the Hessian of the objective at x in sparse format."""
        options = np.zeros(2, dtype=int)
        xz = np.concatenate((x, z))
        if self.__first_sparse_hess_eval:
            nnz, rind, cind, values =  \
                adolc.colpack.sparse_hess_no_repeat(self._lag_trace_id,
                                                    xz, options=options)
            self.nnzH = nnz
            self.hess_rind = rind
            self.hess_cind = cind
            self.hess_values = values
            self.__first_sparse_hess_eval = False

        else:
            nnz, rind, cind, values =  \
                adolc.colpack.sparse_hess_repeat(self._lag_trace_id,
                                                 xz,
                                                 self.hess_rind,
                                                 self.hess_cind,
                                                 self.hess_values)

        # We've computed the Hessian with respect to (x,z).
        # Return only the Hessian with respect to x.
        mask = np.where(rind < self.nvar and cind < self.nvar)
        return (values[mask], rind[mask], cind[mask])

    def jac(self, x, **kwargs):
        """Return constraints Jacobian at x in sparse format."""
        options = np.zeros(4, dtype=int)
        if self.__first_sparse_jac_eval:
            nnz, rind, cind, values =  \
                adolc.colpack.sparse_jac_no_repeat(self._con_trace_id,
                                                   x, options=options)
            self.nnzJ = nnz
            self.jac_rind = rind
            self.jac_cind = cind
            self.jac_values = values
            self.__first_sparse_jac_eval = False

        else:
            nnz, rind, cind, values =  \
                adolc.colpack.sparse_jac_repeat(self._jac_trace_id,
                                                x,
                                                self.jac_rind,
                                                self.jac_cind,
                                                self.jac_values)
        return (values, rind, cind)


class PySparseAdolcModel(PySparseNLPModel, SparseAdolcModel):
    """`AdolcModel` with PySparse sparse matrices."""

    # MRO: 1. PySparseAdolcModel
    #      2. PySparseNLPModel
    #      3. SparseAdolcModel
    #      4. BaseAdolcModel
    #      5. NLPModel
    pass


class SciPyAdolcModel(SciPyNLPModel, SparseAdolcModel):
    """`AdolcModel` with SciPy COO sparse matrices."""

    # MRO: 1. SciPyAdolcModel
    #      2. SciPyNLPModel
    #      3. SparseAdolcModel
    #      4. BaseAdolcModel
    #      5. NLPModel
    pass
