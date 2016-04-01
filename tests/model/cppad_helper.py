from nlp.model.cppadmodel import CppADModel
import numpy as np


class CppADRosenbrock(CppADModel):
    """The standard Rosenbrock function."""

    def obj(self, x, **kwargs):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class CppADHS7(CppADModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return np.array([(1 + x[0]**2)**2 + x[1]**2])
