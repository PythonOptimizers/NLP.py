from nlp.model.cppadmodel import CppADModel
import numpy as np


class CppADHS7(CppADModel):
    """Problem #7 in the Hock and Schittkowski collection."""

    def __init__(self):
        super(CppADHS7, self).__init__(2,
                                       m=1,
                                       x0=np.array([2., 2.]),
                                       Lcon=np.array([0.0]),
                                       Ucon=np.array([0.0]))

    def obj(self, x, **kwargs):
        return np.log(1 + x[0]**2) - x[1]

    def cons(self, x, **kwargs):
        return np.array([(1 + x[0]**2)**2 + x[1]**2])
