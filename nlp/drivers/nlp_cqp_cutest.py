#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nlp_cqp_cutest.py

Solve an NLP from the CUTEst collection using the NLP.py CQP solver.

CQP assumes the problem is convex quadratic. If a general NLP is selected,
CQP will try to minimize the quadratic approximation at the initial point.
"""

from cutest.model.cutestmodel import PySparseCUTEstModel
from nlp.model.pysparsemodel import PySparseSlackModel
from nlp.optimize.cqp import RegQPInteriorPointSolver as CQP
from nlp.tools.logs import config_logger
import numpy as np
import logging
import sys
import argparse

# Set up the problem loggers
def cqp_stats(cqp):
    """Obtain CQP statistics and indicate failures with negatives."""
    print cqp.status
    if cqp.status == "opt":
        it = cqp.iter
        fc, gc = cqp.qp.obj.ncalls, cqp.qp.grad.ncalls
        pg = cqp.kktRes
        ts = cqp.tsolve
    else:
        it = -cqp.iter
        fc, gc = -cqp.qp.obj.ncalls, -cqp.qp.grad.ncalls
        pg = -1.0 if cqp.kktRes is None else -cqp.kktRes
        ts = -1.0 if cqp.tsolve is None else -cqp.tsolve
    return (it, fc, gc, pg, ts)

parser = argparse.ArgumentParser()
parser.add_argument("name_list", nargs='+', help="list of SIF files to process")
parser.add_argument("--use_pc", type=bool, default=True, 
    help="Use Mehrotra's predictor-corrector strategy. If False, use long-step method")
parser.add_argument("--use_scale", type=str, default="none", choices=["none","abs","mc29"],
    help="Choose no scaling (default), scaling based on the Jacobian values (abs), or use MC29 to compute factors (mc29)")
args = parser.parse_args()

nprobs = len(args.name_list)

# Create root logger.
# logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s",
                                stream=None,
                                filename="cutest_qp_rough.txt", filemode="a")

# Create Auglag logger.
cqp_logger = config_logger("nlp.cqp",
                              "%(name)-8s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%9s %5s %5s %6s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "ncon", "iter", "f", u"‖F(w)‖", "#f", u"#∇f", "stat",
                "time")

# Solve problems
for name in args.name_list:
    if name[-4:] == ".SIF":
        name = name[:-4]

    prob = PySparseCUTEstModel(name)
    # prob.compute_scaling_obj()
    # prob.compute_scaling_cons()

    slack_prob = PySparseSlackModel(prob)
    cqp = CQP(slack_prob, mehrotra_pc=args.use_pc, scale_type=args.use_scale)

    # Solve the problem and print the result
    try:
        cqp.solve()
        status = cqp.status
        niter, fcalls, gcalls, kktRes, tsolve = cqp_stats(cqp)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, kktRes, tsolve = cqp_stats(cqp)

    logger.info("%9s %5d %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
            prob.name, prob.nvar, prob.ncon, niter, cqp.qpObj, kktRes,
            fcalls, gcalls, status, tsolve)
