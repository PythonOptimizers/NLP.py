# -*- coding: utf-8 -*-
try:
    from pyomo.environ import *
    from pyomo.opt import ProblemFormat
except ImportError:
    import sys
    print "Pyomo is not installed."
    sys.exit(0)

from nlp.model.amplmodel import AmplModel
import numpy as np


def create_simple_model():
    model = ConcreteModel()
    model.x = Var()
    model.o = Objective(expr=model.x)
    model.c = Constraint(expr=model.x >= 1)
    model.x.set_value(1.0)
    return model

def create_quadratic_model():
    model = ConcreteModel()

    model.x = Var(within=NonNegativeReals)
    model.y = Var(within=NonNegativeReals)

    def constraint_rule(model):
        return model.x + model.y >= 10
    model.c = Constraint(rule=constraint_rule)

    def objective_rule(model):
        return model.x + model.y + 0.5 * (model.x * model.x + 4 * model.x * model.y + 7 * model.y * model.y)
    model.o = Objective(rule=objective_rule, sense=minimize)
    return model

def write_nl(model, nl_filename):
    """Writes a Pyomo model in NL file format."""
    _, smap_id = model.write(nl_filename,
                             format=ProblemFormat.nl)


def display_model(model):
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

if __name__ == "__main__":
    pyomo_simple_model = create_simple_model()
    nl_filename = "simple.nl"
    write_nl(pyomo_simple_model, nl_filename)

    # Create an Amplmodel from the generated nl file
    print '------------------------'
    print 'Problem', nl_filename
    print '------------------------'
    simple_model = AmplModel(nl_filename)
    display_model(simple_model)

    pyomo_quadratic_model = create_quadratic_model()
    nl_filename = "quadratic.nl"
    write_nl(pyomo_quadratic_model, nl_filename)

    # Create an Amplmodel from the generated nl filel
    print '\n\n\n------------------------'
    print 'Problem', nl_filename
    print '------------------------'
    quadratic_model = AmplModel(nl_filename)
    display_model(quadratic_model)

