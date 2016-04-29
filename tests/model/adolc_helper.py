from nlp.model.adolcmodel import AdolcModel, SparseAdolcModel
import numpy as np


class AdolcRosenbrock(AdolcModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class SparseRosenbrock(SparseAdolcModel, AdolcRosenbrock):
    pass


class AdolcHs7(AdolcModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return np.array([(1 + x[0]**2)**2 + x[1]**2])


class SparseHs7(SparseAdolcModel, AdolcHs7):
    pass


try:
    from nlp.model.adolcmodel import PySparseAdolcModel

    class PySparseRosenbrock(PySparseAdolcModel, AdolcRosenbrock):
        pass

    class PySparseHs7(PySparseAdolcModel, AdolcHs7):
        pass

except:
    pass


try:
    from nlp.model.adolcmodel import SciPyAdolcModel

    class SciPyRosenbrock(SciPyAdolcModel, AdolcRosenbrock):
        pass

    class SciPyHs7(SciPyAdolcModel, AdolcHs7):
        pass

except:
    pass
