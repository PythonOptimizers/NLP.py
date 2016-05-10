#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for CQP."""

import logging
import sys
from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.optimize.cqp import RegQPInteriorPointSolver
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger


def cqp_stats(cqp):
    """Obtain CQP statistics and indicate failures with negatives."""
    if cqp.status in ("fatol", "frtol", "gtol"):
        it = cqp.iter
        fc, gc = cqp.model.obj.ncalls, cqp.model.grad.ncalls
        pg = cqp.pgnorm
        ts = cqp.tsolve
    else:
        it = -cqp.iter
        fc, gc = -cqp.model.obj.ncalls, -cqp.model.grad.ncalls
        pg = -1.0 if cqp.pgnorm is None else -cqp.pgnorm
        ts = -1.0 if cqp.tsolve is None else -cqp.tsolve
    return (it, fc, gc, pg, ts)


nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")

# Create CQP logger.
cqp_logger = config_logger("nlp.cqp",
                           "%(name)-8s %(levelname)-5s %(message)s",
                           level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%12s %5s %6s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "iter", "f", u"‖P∇f‖", "#f", u"#∇f", "stat",
                "time")

for problem in sys.argv[1:]:
    model = PySparseAmplModel(problem)
    model.compute_scaling_obj()

    # Check for inequality- or equality-constrained problem.
    if model.m > 0:
        msg = '%s has %d linear or nonlinear constraints'
        logger.error(msg, model.name, model.m)
        continue

    cqp = RegQPInteriorPointSolver(model, TruncatedCG, maxiter=100)
    try:
        cqp.solve()
        status = cqp.status
        niter, fcalls, gcalls, pgnorm, tsolve = cqp_stats(cqp)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, pgnorm, tsolve = cqp_stats(cqp)

    logger.info("%12s %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
                model.name, model.nvar, niter, cqp.f, pgnorm,
                fcalls, gcalls, status, tsolve)
