# -*- coding: utf-8 -*-

"""Test for amplmodel module."""

from nlp.model.amplmodel import AmplModel
import numpy as np
import sys

nargs = len(sys.argv)
if nargs < 1:
    sys.stderr.write('Please specify problem name')
    exit(1)

problem_name = sys.argv[1]

# Create a model
print 'Problem', problem_name
model = AmplModel(problem_name)

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
print ' nnzJ = ', len(jvals)
print ' nnzH = ', len(hvals)

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

print
print ' Testing objective scaling:'
print

print 'Maximum/Minimum gradient (unscaled): %12.5e / %12.5e' \
      % (max(abs(g0)), min(abs(g0)))
model.compute_scaling_obj()  # default is to use x0
g = model.grad(x0)
print 'Maximum/Minimum gradient ( scaled): %12.5e / %12.5e' \
      % (max(abs(g)), min(abs(g)))
model.compute_scaling_obj(reset=True)
g = model.grad(x0)
print '... after a reset ...'
print 'Maximum/Minimum gradient (unscaled): %12.5e / %12.5e' \
      % (max(abs(g)), min(abs(g)))

print
print ' Testing constraint scaling:'
print

for i in xrange(min(ncon, 5)):
    model.compute_scaling_cons(reset=True)
    sgi = model.sigrad(i, x0)
    imax = max(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    imin = min(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    print 'Constraint %3i: ' % i,
    print ' Max/Min gradient (unscaled): %12.5e (%3i) / %12.5e (%3i)' \
        % (sgi[imax], imax, sgi[imin], imin)
    model.compute_scaling_cons()
    sgi = model.sigrad(i, x0)
    imax = max(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    imin = min(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    print 'Constraint %3i: ' % i,
    print ' Max/Min gradient ( scaled): %12.5e (%3i) / %12.5e (%3i)' \
        % (sgi[imax], imax, sgi[imin], imin)

# Output "solution"
model.writesol(x0, pi0, 'And the winner is')
