#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple AMPL driver for TRON."""

import sys
from nlp.optimize.tron import TRON
from nlp.tools.logs import config_logger
from nlp.model.cysparsemodel import CySparseAmplModel
import logging

nprobs = len(sys.argv) - 1
if nprobs == 0:
    raise ValueError("Please supply problem name as argument")

# Create root logger.
logger = config_logger("nlp", "%(name)-9s %(levelname)-5s %(message)s")

# Create TRON logger.
tron_logger = config_logger("tron",
                            "%(name)-9s %(levelname)-5s %(message)s",
                            level=logging.WARN if nprobs > 1 else logging.INFO)
if nprobs > 1:
    logger.info("%10s %5s %5s %8s %7s %5s %5s %4s %s",
                "name", "nvar", "#iter", "f", u"‖∇f‖", "#f", u"#∇f", "stat",
                "time")

for problem in sys.argv[1:]:
    model = CySparseAmplModel(problem)

    # Check for inequality- or equality-constrained problem.
    if model.m > 0:
        msg = '%s has %d linear or nonlinear constraints'
        logger.error(msg % (model.name, model.m))
        error = True
        continue

    tron = TRON(model, logger_name="tron")
    tron.solve()

    if nprobs > 1:
        logger.info("%10s %5d %5d %8.1e %7.1e %5d %5d %4s %.3f",
                    model.name, model.nvar, tron.iter, tron.f, tron.pgnorm,
                    model.obj.ncalls, model.grad.ncalls,
                    tron.status, tron.tsolve)
    else:
        # Output final statistics
        logger.info('--------------------------------')
        logger.info('TRON: End of Execution')
        logger.info('  Problem                      : %-s' % model.name)
        logger.info('  Number of variables          : %-d' % model.nvar)
        logger.info('  Initial/Final Objective      : %-g/%-g' % (tron.f0, tron.f))
        logger.info('  Initial/Final Projected Gradient Norm : %-g/%-g' %
                    (tron.pg0, tron.pgnorm))
        logger.info('  Number of iterations         : %-d' % tron.iter)
        logger.info('  Number of function evals     : %-d' % model.obj.ncalls)
        logger.info('  Number of gradient evals     : %-d' % model.grad.ncalls)
        logger.info('  Number of CG iterations      : %-d' % tron.total_cgiter)
        logger.info('  Solve time                   : %-gs' % tron.tsolve)
        logger.info('  Status                       : %-s' % tron.status)
        logger.info('--------------------------------')
