#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for Auglag."""

import logging
import sys
from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.optimize.auglag import Auglag
from nlp.tools.logs import config_logger
from nlp.optimize.tron import TRON


def auglag_stats(auglag):
    """Obtain Auglag statistics and indicate failures with negatives."""
    print auglag.status
    if auglag.status == "opt":
        it = auglag.iter
        fc, gc = auglag.model.model.obj.ncalls, auglag.model.model.grad.ncalls
        pg = auglag.pgnorm
        ts = auglag.tsolve
    else:
        it = -auglag.iter
        fc, gc = -auglag.model.obj.ncalls, -auglag.model.grad.ncalls
        pg = -1.0 if auglag.pgnorm is None else -auglag.pgnorm
        ts = -1.0 if auglag.tsolve is None else -auglag.tsolve
    return (it, fc, gc, pg, ts)


nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")

# Create Auglag logger.
auglag_logger = config_logger("nlp.auglag",
                              "%(name)-8s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.INFO)
# Create TRON logger.
tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%12s %5s %6s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "iter", "f", u"‖P∇f‖", "#f", u"#∇f", "stat",
                "time")

for problem in sys.argv[1:]:
    model = PySparseAmplModel(problem)
    model.compute_scaling_obj()

    print model.x0
    # model.compute_scaling_obj()

    auglag = Auglag(model, TRON, maxiter=100)
    # try:
    auglag.solve()
    status = auglag.status
    niter, fcalls, gcalls, pgnorm, tsolve = auglag_stats(auglag)
    # except:
    #     msg = sys.exc_info()[1].message
    #     status = msg if len(msg) > 0 else "xfail"  # unknown failure
    #     print "here"
    #     niter, fcalls, gcalls, pgnorm, tsolve = auglag_stats(auglag)

    logger.info("%12s %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
                model.name, model.nvar, niter, auglag.f, pgnorm,
                fcalls, gcalls, status, tsolve)
