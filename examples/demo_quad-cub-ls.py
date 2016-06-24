# -*- coding: utf8 -*-
"""Demo for Quadratic/Cubic linesearch."""
from nlp.model.linemodel import C1LineModel
from nlp.ls.quad_cub import QuadraticCubicLineSearch
from nlp.tools.exceptions import LineSearchFailure
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

this_path = os.path.dirname(os.path.realpath(__file__))

lib_path = os.path.abspath(os.path.join(
    this_path, '..', '..', '..', 'tests', 'model'))
sys.path.append(lib_path)

from python_models import SimpleQP, SimpleCubicProb
from python_models import LineSearchProblem, LineSearchProblem2


def quadratic_interpolant(ls, t):
    u"""Build quadratic interpolant of ϕ with ϕ(t₀), ϕ'(0) and ϕ(0).

    |   ϕq(t) := t²(ϕ(t₀) - ϕ(0) - t₀ϕ'(0))/t₀² + ϕ'(0)t + ϕ(0)
    """
    slope = ls.slope
    phi0 = ls.value
    phi_t0 = ls.linemodel.obj(ls._step0)

    phi = np.empty(t.size)
    phi = t**2 * (phi_t0 - phi0 - ls._step0 * slope) / \
        ls._step0**2 + slope * t + phi0

    return phi


def cubic_interpolant(ls, t, a0, a1):
    u"""Build quadratic interpolant of ϕ with ϕ(t₀), ϕ'(0) and ϕ(0).

    |   ϕc(t) := a t³ + b t² + ϕ'(0) t + ϕ(0)
    """
    try:
        a0 = a0[0]
        a1 = a1[0]
    except:
        pass
    slope = ls.slope
    phi0 = ls.value

    phi_a0 = ls.linemodel.obj(a0)
    phi_a1 = ls.linemodel.obj(a1)
    c = 1. / (a0**2 * a1**2 * (a1 - a0))
    M = np.array([[a0**2, -a1**2], [-a0**3, a1**3]])
    coeffs = c * np.dot(M, np.array([phi_a1 - phi0 - slope * a1,
                                     phi_a0 - phi0 - slope * a0]))

    phic = coeffs[0] * t**3 + coeffs[1] * t**2 + t * slope + phi0
    return phic


def run_demo(model, x, p, step0):
    """Create arrays necessary to display steps of linesearch."""
    c1model = C1LineModel(model, x, p)
    ls = QuadraticCubicLineSearch(c1model, step=step0)

    t = np.linspace(-0.2, 1.2 * ls._step0, 1000)

    y = np.empty(t.size)
    k = 0
    for x in t:
        y[k] = c1model.obj(x)
        k += 1
    plt.figure()
    plt.ion()
    plt.plot(t, y)
    t = np.linspace(0, ls._step0, 1000)

    x_p = []
    y_p = []
    x_p.append(0.)
    y_p.append(ls._value)
    plt.annotate(
        "$t=0$",
        xy=(0, ls._value), xytext=(-5, 5),
        textcoords='offset points', ha='right', va='bottom')

    x_p.append(ls.step)
    y_p.append(ls.trial_value)
    plt.scatter(x_p, y_p)

    plt.annotate(
        "$t_0=" + str(ls.step) + "$",
        xy=(ls.step, ls.trial_value), xytext=(-5, 5),
        textcoords='offset points', ha='right', va='bottom')

    try:
        for k, step in enumerate(ls):
            print k, step
            if k == 0:
                phi = quadratic_interpolant(ls, t)
                curve, = plt.plot(t, phi)
                last_step = ls._last_step
                s = ls.step
            else:
                phi3 = cubic_interpolant(ls, t, last_step, s)
                curve, = plt.plot(t, phi3)
                last_step = ls._last_step
                s = ls.step
            x_p.append(ls.step)
            y_p.append(ls.trial_value)
            plt.annotate(
                "$t_" + str(k + 1) + "=%3.1f" % ls.step + "$",
                xy=(ls.step, ls.trial_value), xytext=(-5, 5),
                textcoords='offset points', ha='right', va='bottom')
            plt.scatter(x_p, y_p)
            plt.pause(1)
            curve.remove()
    except LineSearchFailure:
        pass

model = SimpleQP()
x = np.ones(model.nvar)
run_demo(model, x, -model.grad(x), step0=1)

run_demo(model, x, -model.grad(x), step0=2)

model = SimpleCubicProb()
x = np.ones(model.nvar)
run_demo(model, x, -model.grad(x), step0=1)

run_demo(LineSearchProblem(), np.array([3.5]), np.array([-1]), step0=3)

run_demo(LineSearchProblem(), np.array([-0.5]), np.array([1]), step0=4)

run_demo(LineSearchProblem2(), np.array([-4.]), np.array([1]), step0=8)
