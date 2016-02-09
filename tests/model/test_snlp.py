# Tests relative to algorithmic differentiation with AMPL.
# from nlp.model.amplpy import AmplModel
from nlp.model.cysparsemodel import CySparseAmplModel, CySparseSlackModel
from nlp.model.pysparsemodel import PySparseAmplModel, PySparseSlackModel
from helper import *
import numpy as np
import os

this_path = os.path.dirname(os.path.realpath(__file__))

class Hs7Data(object):

  def __init__(self):
    self.expected_f = -0.39056208756589972
    self.expected_c = np.array([25.0])
    self.expected_l = -25.3905620876   # uses cons_pos().
    self.expected_g = np.array([0.8, -1.])
    self.expected_H = np.array([[-52.24,  0.],
                                [  0.  , -2.]])
    v = np.arange(1, self.expected_H.shape[0] + 1, dtype=np.float)
    self.expected_Hv = np.dot(self.expected_H, v)
    self.expected_J = np.array([[40., 4.]])
    self.expected_Jv = np.dot(self.expected_J, v)
    w = 2 * np.ones(self.expected_J.shape[0])
    self.expected_JTw = np.dot(self.expected_J.T, w)


class Hs7(object):

  def get_expected(self):
    return Hs7Data()

  def get_derivatives(self, model):
    return get_derivatives_coord(model)

  def test_hs7(self):
    hs7_data = self.get_expected()
    (f, c, l) = get_values(self.model)
    assert_almost_equal(f, hs7_data.expected_f)
    assert_allclose(c, hs7_data.expected_c)
    assert_almost_equal(l, hs7_data.expected_l)
    (g, H, Hv, J, Jv, JTw) = self.get_derivatives(self.model)
    assert(np.allclose(g, hs7_data.expected_g))
    assert(np.allclose(H, hs7_data.expected_H))
    assert(np.allclose(Hv, hs7_data.expected_Hv))
    assert(np.allclose(J, hs7_data.expected_J))
    assert(np.allclose(Jv, hs7_data.expected_Jv))
    assert(np.allclose(JTw, hs7_data.expected_JTw))


class Test_SlackModelPySparseAmplHS7(TestCase, Hs7):    # Test defined in Hs7

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        model = os.path.join(this_path, 'hs007.nl')
        model = PySparseAmplModel(model)
        model.pi0 = np.ones(1)
        self.model = PySparseSlackModel(model)    # x0 = (2, 2)


# class Test_SlackModelCySparseAmplHS7(TestCase, Hs7):    # Test defined in Hs7
#
#     def get_derivatives(self, model):
#         return get_derivatives_llmat(model)
#
#     def setUp(self):
#         model = os.path.join(this_path, 'hs007.nl')
#         model = CySparseAmplModel(model)
#         model.pi0 = np.ones(1)
#         self.model = CySparseSlackModel(model)    # x0 = (2, 2)


if __name__ == '__main__':

    import unittest
    unittest.main()
