"""Tests relative to problems modeled with AMPL."""

from unittest import TestCase
import numpy as np
from helper import *
import os
import pytest
from nlp.tools.dercheck import DerivativeChecker
from nlp.tools.logs import config_logger
import logging

this_path = os.path.dirname(os.path.realpath(__file__))


class Test_AmplRosenbrock(TestCase, Rosenbrock):  # Test also def'd in Rosenbrock

    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = AmplModel(model)  # x0 = (-1, ..., -1)

    def test_ampl_hessian_issue(self):
        data = self.get_expected()
        model = self.model

        # Trying to trick AMPL.
        f = model.obj(np.zeros(self.model.n))
        H_lag = ndarray_from_coord(model.nvar, model.nvar,
                                   *model.hess(model.x0, model.pi0),
                                   symmetric=True)
        assert np.allclose(H_lag, data.expected_H_lag)

        # Trying to trick AMPL.
        f = model.obj(np.zeros(self.model.n))
        v = np.arange(1, model.nvar + 1, dtype=np.float)
        Hv = model.hprod(model.x0, model.pi0, v)
        assert np.allclose(Hv, data.expected_Hv)


class Test_AmplHS7(TestCase, Hs7):  # Test also defined in Hs7

    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = AmplModel(model)  # x0 = (2, 2)
        self.model.pi0 = np.ones(1)


class Test_AmplMaxProfit(TestCase, MaxProfit):  # Test also defined in MaxProfit

    def get_derivatives(self, model):
        return get_derivatives_coord(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'max_profit.mod')
        self.model = AmplModel(model)  # x0 = (0, 0)
        total_m = self.model.nlowerC + self.model.nupperC + \
            2 * self.model.nrangeC
        total_m += self.model.nlowerB + self.model.nupperB + \
            2 * self.model.nrangeB
        self.model.pi0 = np.ones(total_m)


class Test_PySparseAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = PySparseAmplModel(model)    # x0 = (-1, ..., -1)


class Test_PySparseAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_PySparseAmplHS9(TestCase, Hs9):

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs009.nl')
        self.model = PySparseAmplModel(model)
        self.model.x0 = np.ones(2)
        self.model.pi0 = np.ones(1)


class Test_PySparseSlackHS7(TestCase, GenericTest):

    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(1)

    def test_init(self):
        model = os.path.join(this_path, 'hs007.nl')
        with pytest.raises(TypeError):
            model = PySparseSlackModel(AmplModel(model))


class Test_PySparseSlackHS9(TestCase, GenericTest):

    def get_expected(self):
        return Hs9SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs009.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.x0 = np.ones(2)
        self.model.pi0 = np.ones(1)

    def test_lp_jac(self):
        assert np.allclose(self.model._jac(self.model.x0, lp=True).getNumpyArray(),
                           np.array([[4., -3.]]))


class Test_PySparseSlackHS10(TestCase, GenericTest):

    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_llmat(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("pysparse")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = PySparseSlackModel(PySparseAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1


class Test_SciPyAmplRosenbrock(TestCase, Rosenbrock):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = SciPyAmplModel(model)  # x0 = (-1, ..., -1)


class Test_SciPyAmplHS7(TestCase, Hs7):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPyAmplModel(model)
        self.model.pi0 = np.ones(1)


class Test_SciPyAmplHS9(TestCase, Hs9):

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs009.nl')
        self.model = SciPyAmplModel(model)
        self.model.x0 = np.ones(2)
        self.model.pi0 = np.ones(1)


class Test_SciPySlackRosenbrock(TestCase, GenericTest):

    def get_expected(self):
        return RosenbrockData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'rosenbrock.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))


class Test_SciPySlackHS7(TestCase, GenericTest):

    def get_expected(self):
        return Hs7SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs007.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(1)


class Test_SciPySlackHS9(TestCase, GenericTest):

    def get_expected(self):
        return Hs9SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs009.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.x0 = np.ones(2)
        self.model.pi0 = np.ones(1)

    def test_lp_jac(self):
        assert np.allclose(self.model._jac(self.model.x0, lp=True).todense(),
                           np.array([[4., -3.]]))


class Test_SciPySlackHS10(TestCase, GenericTest):

    def get_expected(self):
        return Hs10SlackData()

    def get_derivatives(self, model):
        return get_derivatives_scipy(model)

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        pytest.importorskip("scipy")
        model = os.path.join(this_path, 'hs010.nl')
        self.model = SciPySlackModel(SciPyAmplModel(model))
        self.model.pi0 = np.ones(2)
        self.model.x0[2] = 1


class Test_GenTemplate(TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        self.model_name = os.path.join(this_path, 'hs007.mod')

    def test_template_creation(self):
        model = AmplModel(self.model_name)
        assert model.name == "hs007"
        with pytest.raises(ValueError):
            model = AmplModel("sylvain.nl")

    def test_obj_scaling(self):
        model = AmplModel(self.model_name)
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        model.compute_scaling_obj(g_max=1.)
        assert model.obj(model.x0) == -0.39056208756589972
        assert np.allclose(model.grad(model.x0), np.array([0.8, -1.]))
        assert model.scale_obj == 1.

        model.compute_scaling_obj(reset=True)
        assert model.scale_obj is None

        model.compute_scaling_obj(g_max=0.5)
        assert model.obj(model.x0) == -0.19528104378294986
        assert np.allclose(model.grad(model.x0), np.array([0.4, -0.5]))
        assert np.allclose(model.sgrad(model.x0).to_array(),
                           np.array([0.4, -0.5]))
        assert model.scale_obj == 0.5

        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

    def test_cons_scaling(self):
        model = AmplModel(self.model_name)
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.jac_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        model.compute_scaling_cons(g_max=40.)
        assert np.allclose(model.scale_con, np.array([1.]))
        assert np.allclose(model.cons(model.x0), np.array([29.0]))
        assert np.allclose(model.jop(model.x0).to_array(),
                           np.array([[40., 4.]]))
        model.compute_scaling_cons(reset=True)
        assert model.scale_con is None

        model.compute_scaling_cons(g_max=20.)
        assert np.allclose(model.cons(model.x0), np.array([14.5]))
        assert np.allclose(model.scale_con, np.array([0.5]))
        assert np.allclose(model.jop(model.x0).to_array(),
                           np.array([[20., 2.]]))
        assert np.allclose(model.igrad(0, model.x0),
                           np.array([20., 2.]))
        assert np.allclose(model.sigrad(0, model.x0).to_array(),
                           np.array([20., 2.]))

        v = np.ones(model.nvar)
        assert np.allclose(model.hiprod(model.x0, 0, v),
                           np.array([26., 1.]))

        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.jac_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        print model.display_basic_info()


class Test_AmplModelExtrasim(TestCase):

    def setUp(self):
        pytest.importorskip("nlp.model.amplmodel")
        model = os.path.join(this_path, 'extrasim.nl')
        self.model = AmplModel(model)

    def test_lp(self):
        model = self.model
        assert model.islp() is True
        assert np.allclose(model.cost().to_array(), np.array([1., 0.]))
        print model.display_basic_info()

    def test_obj_scaling(self):
        model = self.model
        model.x0 = np.zeros(model.nvar)

        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        model.compute_scaling_obj(g_max=1.)
        assert model.scale_obj == 1.
        assert model.obj(model.x0) == 1.
        assert np.allclose(model.grad(model.x0), np.array([1., 0.]))
        print model.cost()
        assert np.allclose(model.cost().to_array(), np.array([1., 0.]))

        model.compute_scaling_obj(reset=True)
        assert model.scale_obj is None

        model.compute_scaling_obj(g_max=0.5)
        assert model.scale_obj == 0.5
        assert model.obj(model.x0) == 0.5
        assert np.allclose(model.grad(model.x0), np.array([0.5, 0.]))
        assert np.allclose(model.cost().to_array(), np.array([0.5, 0.]))
        assert np.allclose(model.sgrad(model.x0).to_array(),
                           np.array([0.5, 0.]))

        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.grad_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

    def test_icons(self):
        model = self.model
        cons = np.array([0.])
        for j in range(model.ncon):
            assert np.allclose(model.icons(j, model.x0), cons[j])

    def test_irow(self):
        model = self.model
        A = np.array([[1., 2.]])
        for j in range(model.ncon):
            assert np.allclose(model.irow(j).to_array(), A[j, :])

    def test_cons_scaling(self):
        model = self.model
        log = config_logger("nlp.der",
                            "%(name)-10s %(levelname)-8s %(message)s",
                            level=logging.DEBUG)
        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.jac_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        model.compute_scaling_cons(g_max=2.)
        assert np.allclose(model.scale_con, np.array([1.]))
        cons = np.array([0.])
        for j in range(model.ncon):
            assert np.allclose(model.icons(j, model.x0), cons[j])
        assert np.allclose(model.cons(model.x0), cons)
        assert np.allclose(model.jop(model.x0).to_array(),
                           np.array([[1., 2.]]))
        assert np.allclose(ndarray_from_coord(model.ncon, model.nvar,
                                              *model.A(), symmetric=False),
                           np.array([[1., 2.]]))

        model.compute_scaling_cons(reset=True)
        assert model.scale_con is None

        model.compute_scaling_cons(g_max=1.)
        assert np.allclose(model.scale_con, np.array([0.5]))
        assert np.allclose(model.cons(model.x0), np.array([0.]))
        assert np.allclose(model.jop(model.x0).to_array(),
                           np.array([[0.5, 1.]]))
        assert np.allclose(ndarray_from_coord(model.ncon, model.nvar,
                                              *model.A(), symmetric=False),
                           np.array([[0.5, 1.]]))
        assert np.allclose(model.igrad(0, model.x0),
                           np.array([0.5, 1.]))
        assert np.allclose(model.sigrad(0, model.x0).to_array(),
                           np.array([0.5, 1.]))

        v = np.ones(model.nvar)
        assert np.allclose(model.hiprod(model.x0, 0, v),
                           np.zeros(model.n))

        dcheck = DerivativeChecker(model, model.x0, tol=1e-5)
        dcheck.check(hess=True, chess=True)
        assert len(dcheck.jac_errs) == 0
        assert len(dcheck.hess_errs) == 0
        for j in xrange(model.ncon):
            assert (len(dcheck.chess_errs[j]) == 0)

        print model.display_basic_info()
