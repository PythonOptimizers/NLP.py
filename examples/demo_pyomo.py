# -*- coding: utf-8 -*-
try:
    from pyomo.environ import *
    from pyomo.opt import ProblemFormat
except ImportError:
    import sys
    print "Pyomo is not installed."
    sys.exit(0)

from nlp.model.amplmodel import AmplModel
from nlp.optimize.tron import TRON
from nlp.optimize.pcg import TruncatedCG
from nlp.tools.logs import config_logger
import logging

import numpy as np

tron_logger = config_logger("nlp.tron",
                            "%(name)-8s %(levelname)-5s %(message)s",
                            level=logging.INFO)

def create_simple_model():
    model = ConcreteModel()
    model.x = Var(bounds=(1., None))
    model.o = Objective(expr=model.x)
    model.x.set_value(20.0)
    return model

def create_rosenbrock_model():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var(bounds=(-1.5, None))
    model.o = Objective(expr=(model.x-1)**2 + \
                              100*(model.y-model.x**2)**2)
    model.x.set_value(-2.)
    model.y.set_value(1.)
    return model

def write_nl(model, nl_filename):
    """Writes a Pyomo model in NL file format."""
    _, smap_id = model.write(nl_filename,
                             format=ProblemFormat.nl)

def display_model(model):
    
    print '------------------------'
    print 'Model: %15s' % model.name
    print '------------------------'

    # Query the model
    x0 = model.x0
    pi0 = model.pi0
    nvar = model.nvar
    ncon = model.ncon
    print 'There are %d variables and %d constraints' % (nvar, ncon)
    
    np.set_printoptions(precision=3, linewidth=79, threshold=10, edgeitems=3)
    
    print 'Initial point: ', x0
    print 'Lower bounds on x: ', model.Lvar
    print 'Upper bounds on x: ', model.Uvar
    print 'f(x0) = ', model.obj(x0)
    g0 = model.grad(x0)
    print '∇f(x0) = ', g0
    
    if ncon > 0:
        print 'Initial multipliers: ', pi0
        print 'Lower constraint bounds: ', model.Lcon
        print 'Upper constraint bounds: ', model.Ucon
        print 'c(x0) = ', model.cons(x0)
    
    jvals, jrows, jcols = model.jac(x0)
    hvals, hrows, hcols = model.hess(x0, pi0)
    print
    print 'nnzJ = ', len(jvals)
    print 'nnzH = ', len(hvals)

    print 'J(x0) = (in coordinate format)'
    print 'vals: ', jvals
    print 'rows: ', jrows
    print 'cols: ', jcols
    print 'Hessian (lower triangle in coordinate format):'
    print 'vals: ', hvals
    print 'rows: ', hrows
    print 'cols: ', hcols
    
    if ncon > 0:
        print
        print ' Evaluating constraints individually, sparse gradients'
        print
    
    for i in range(min(ncon, 5)):
        ci = model.icons(i, x0)
        print 'c%d(x0) = %-g' % (i, ci)
        sgi = model.sigrad(i, x0)
        k = sgi.keys()
        ssgi = {}
        for j in range(min(5, len(k))):
            ssgi[k[j]] = sgi[k[j]]
        print '∇c%d(x0) = ' % i, ssgi
    
    print
    print ' Testing matrix-vector product:'
    print
    
    e = np.ones(nvar)
    He = model.hprod(x0, pi0, e)
    print 'He = ', He
    print '\n\n\n'

if __name__ == "__main__":
    pyomo_simple_model = create_simple_model()
    nl_filename = "simple.nl"
    write_nl(pyomo_simple_model, nl_filename)

    # Create an Amplmodel from the generated nl file
    simple_model = AmplModel(nl_filename)
    display_model(simple_model)
    
    pyomo_rosenbrock_model = create_rosenbrock_model()
    nl_filename = "rosenbrock.nl"
    write_nl(pyomo_rosenbrock_model, nl_filename)

    # Create an Amplmodel from the generated nl file
    rosenbrock_model = AmplModel(nl_filename)
    display_model(rosenbrock_model)

    solver = TRON(rosenbrock_model, TruncatedCG)
    solver.solve()

