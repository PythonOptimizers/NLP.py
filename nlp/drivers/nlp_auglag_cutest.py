#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nlp_auglag_cutest.py

Solve an NLP from the CUTEst collection using the NLP.py Auglag solver.

Depending on the settings, Auglag may use the full Hessian or a (structured) 
approximation of the Hessian.
"""

from cutest.model.cutestmodel import CUTEstModel, QNCUTEstModel
from nlp.optimize.auglag import Auglag
from nlp.optimize.tron import TRON, QNTRON, StructQNTRON
from pykrylov.linop.lbfgs import LBFGSOperator, CompactLBFGSOperator, StructuredLBFGSOperator
from pykrylov.linop.lsr1 import LSR1Operator, CompactLSR1Operator, StructuredLSR1Operator
from nlp.tools.logs import config_logger
import numpy as np
import logging
import sys
import argparse

# Set up the problem loggers
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

# Argument parsing, take list of names from the command line
parser = argparse.ArgumentParser()
parser.add_argument("name_list", nargs='+', help="list of SIF files to process")
parser.add_argument("--qn_type", type=str, default="bfgs", choices=["bfgs","sr1"],
    help="the type of quasi-Newton approximation to be used")
parser.add_argument("--approx_type", type=str, default=None, choices=[None,"full","struct"],
    help="choose exact Hessian (None), full approximate Hessian (full), or structured approximation (struct)")
args = parser.parse_args()

nprobs = len(args.name_list)

# Create root logger.
# logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s",
                                stream=None,
                                filename="cutest_rough.txt", filemode="a")

# Create Auglag logger.
auglag_logger = config_logger("nlp.auglag",
                              "%(name)-8s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.INFO)

# Create TRON logger.
tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%9s %5s %5s %6s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "ncon", "iter", "f", u"‖P∇L‖", "#f", u"#∇f", "stat",
                "time")

# Solve problems
for name in args.name_list:
    if name[-4:] == ".SIF":
        name = name[:-4]

    if args.approx_type == "struct":
        if args.qn_type == "bfgs":
            prob = QNCUTEstModel(name, H=StructuredLBFGSOperator)
        elif args.qn_type == "sr1":
            prob = QNCUTEstModel(name, H=StructuredLSR1Operator)
    else:
        prob = CUTEstModel(name)

    prob.compute_scaling_obj()
    prob.compute_scaling_cons()

    if args.approx_type == "struct":
        auglag = Auglag(prob, StructQNTRON, maxupdate=100)
    elif args.approx_type == "full":
        if args.qn_type == 'bfgs':
            auglag = Auglag(prob, QNTRON, full_qn=True, H=CompactLBFGSOperator, maxupdate=100)
        elif args.qn_type == 'sr1':
            auglag = Auglag(prob, QNTRON, full_qn=True, H=CompactLSR1Operator, maxupdate=100)
    else:
        auglag = Auglag(prob, TRON, maxupdate=100)  # Exact Hessian

    # Solve the problem and print the result
    try:
        auglag.solve()
        status = auglag.status
        niter, fcalls, gcalls, pgnorm, tsolve = auglag_stats(auglag)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, pgnorm, tsolve = auglag_stats(auglag)

    logger.info("%9s %5d %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
            prob.name, prob.nvar, prob.ncon, niter, auglag.f, pgnorm,
            fcalls, gcalls, status, tsolve)
