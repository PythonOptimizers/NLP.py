#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for Auglag."""

import logging
import os
from argparse import ArgumentParser

from nlp.tools.logs import config_logger
from nlp.optimize.tron import TRON


def auglag_stats(auglag):
    """Obtain Auglag statistics and indicate failures with negatives."""
    print auglag.status
    if auglag.status == "opt":
        it = auglag.iter
        fc = auglag.model.model.model.obj.ncalls
        gc = auglag.model.model.model.grad.ncalls
        hc = auglag.model.model.model.hess.ncalls
        pg = auglag.pgnorm
        if auglag.model.model.model.m > 0:
            cnorm = auglag.cons_norm
        else:
            cnorm = 0.
        jprod = auglag.model.model.model.jprod.ncalls + \
            auglag.model.model.model.jtprod.ncalls
        ts = auglag.tsolve
    else:
        it = -auglag.iter
        fc = -auglag.model.model.model.obj.ncalls
        gc = -auglag.model.model.model.grad.ncalls
        hc = auglag.model.model.model.hess.ncalls
        pg = -1.0 if auglag.pgnorm is None else -auglag.pgnorm
        cnorm = -1.0 if auglag.cons_norm is None else -auglag.cons_norm
        jprod = -(auglag.model.model.model.jprod.ncalls +
                  auglag.model.model.model.jtprod.ncalls)
        ts = -1.0 if auglag.tsolve is None else -auglag.tsolve
    return (it, fc, gc, pg, cnorm, jprod, hc, ts)

desc = "Augmented Lagrangian method. "
desc += "By default, exact second derivatives are used."

# Define allowed command-line options.
parser = ArgumentParser(description=desc)
parser.add_argument("-b", "--bfgs", action="store_true", dest="bfgs",
                    default=False,
                    help="use limited-memory BFGS approximation")
parser.add_argument("-p", "--pairs", type=int,
                    default=5, dest="npairs", help="quasi-Newton memory")
parser.add_argument("-i", "--iter", type=int,
                    default=100, dest="maxiter",
                    help="maximum number of iterations")

# Parse command-line arguments.
(args, other) = parser.parse_known_args()
opts = {}

# Import appropriate components.
quasi_newton = False
if args.bfgs:
    quasi_newton = True
    from nlp.optimize.regsqp.counterfeitamplmodel import QNCounterFeitAmplModel as Model
    from nlp.optimize.tron import QNTRON as TRON
    from nlp.optimize.auglag import QNAuglag as Auglag

    from pykrylov.linop import CompactLBFGSOperator as QNOperator
    opts["H"] = QNOperator
    opts["npairs"] = args.npairs
    opts["scaling"] = True
else:
    from nlp.optimize.tron import TRON
    from nlp.model.pysparsemodel import PySparseAmplModel as Model
    from nlp.optimize.auglag import Auglag

nprobs = len(other)
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")


# Create root logger.
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")

# Create Auglag logger.
auglag_logger = config_logger("nlp.auglag",
                              "%(name)-8s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.DEBUG)
# Create TRON logger.
tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%12s %5s %6s %8s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "iter", "f", u"‖P∇L‖", u"‖c‖",
                "#f", u"#g", "stat", "time")

for problem in other:
    model = Model(problem, **opts)

    prob_name = os.path.basename(problem)
    if prob_name[-3:] == '.nl':
        prob_name = prob_name[:-3]

    print model.x0
    # model.compute_scaling_obj()

    auglag = Auglag(model, TRON, maxiter=args.maxiter, **opts)
    # try:
    auglag.solve()
    status = auglag.status

    niter, fcalls, gcalls, pgnorm, cnorm, jprod, hcalls, tsolve = \
        auglag_stats(auglag)
    # except:
    #     msg = sys.exc_info()[1].message
    #     status = msg if len(msg) > 0 else "xfail"  # unknown failure
    #     print "here"
    #     niter, fcalls, gcalls, pgnorm, tsolve = auglag_stats(auglag)
    print auglag_stats(auglag)
    logger.info("%12s %5d %6d %8.1e %8.1e %8.1e %6d %6d %5s %7.3f",
                model.name, model.nvar, niter, auglag.f, pgnorm, cnorm,
                fcalls,
                gcalls,
                status,
                tsolve)

if nprobs == 1:
    print auglag.x
    # Output final statistics
    logger.info('--------------------------------')
    logger.info('auglag: End of Execution')
    logger.info('  Problem                      : %-s', prob_name)
    logger.info('  Number of variables          : %-d', model.n)
    logger.info('  Number of linear constraints : %-d', model.nlin)
    logger.info('  Number of general constraints: %-d',
                model.m - model.nlin)
    logger.info('  Initial/Final Objective      : %-g/%-g', auglag.f0, auglag.f)
    logger.info('  Number of iterations         : %-d', niter)
    logger.info('  Number of function evals     : %-d', fcalls)
    logger.info('  Number of gradient evals     : %-d', gcalls)
    logger.info('  Number of Jacobian evals     : %-d',
                0 if quasi_newton else gcalls)
    logger.info('  Number of Jacobian products  : %-d',
                jprod if quasi_newton else 0)
    logger.info('  Number of Hessian evals      : %-d',
                0 if quasi_newton else hcalls)
    logger.info('  Solve time                   : %-gs', tsolve)
    logger.info('--------------------------------')
