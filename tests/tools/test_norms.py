from unittest import TestCase
import numpy as np
import pytest

from pykrylov.linop import LinearOperator
from nlp.tools.norms import *


class Test_norm(TestCase):

    def setUp(self):
        self.x = np.array([-1., 3])
        self.A = np.array([[-1, 3.],
                           [4, -2.]])

    def test_norm1(self):
        assert norm1(self.x) == 4.
        assert np.allclose(norm1(self.A), np.array([5., 5.]))

        assert norm1(np.array([])) == 0.0
        assert norm1([]) == 0.0

    def test_norm2(self):
        assert norm2(self.x) == np.sqrt(10.)
        assert norm2(self.A) == np.sqrt(30.)

        assert norm2(np.array([])) == 0.0
        assert norm2([]) == 0.0

    def test_normp(self):
        assert normp(self.x, 2) == np.sqrt(10.)

        assert normp(np.array([]), 2) == 0.0
        assert normp([], 2) == 0.0

    def test_norm_infty(self):
        assert norm_infty(self.x) == 3.
        assert norm_infty(self.A) == 6.

        assert norm_infty(np.array([])) == 0.0
        assert norm_infty([]) == 0.0

    def test_norm_spec(self):
        Aop = LinearOperator(self.A.shape[1], self.A.shape[0],
                             lambda v: np.dot(self.A, v),
                             matvec_transp=lambda v: np.dot(self.A.T, v))
        U, s, V = np.linalg.svd(self.A, full_matrices=True)
        np.testing.assert_approx_equal(normest(Aop)[0], max(s))

        with pytest.raises(Warning):
            v = normest(Aop, maxits=1)
