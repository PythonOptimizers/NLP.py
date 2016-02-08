#!/usr/bin/env python
#
# Test for amplpy module
#
from nlpy.model.pysparsemodel import PySparseAmplModel
import numpy
import getopt
import sys

PROGNAME = sys.argv[0]


def commandline_err(msg):
    sys.stderr.write("%s: %s\n" % (PROGNAME, msg))
    sys.exit(1)


def parse_cmdline(arglist):
    if len(arglist) != 1:
        commandline_err('Specify file name (look in data directory)')
        return None

    try: options, fname = getopt.getopt(arglist, '')

    except getopt.error, e:
        commandline_err("%s" % str(e))
        return None

    return fname[0]

ProblemName = parse_cmdline(sys.argv[1:])

# Create a model
print 'Problem', ProblemName
nlp = PySparseAmplModel(ProblemName)

# Query the model
x0  = nlp.x0
pi0 = nlp.pi0
n = nlp.n
m = nlp.m
print 'There are %d variables and %d constraints' % (n, m)

max_n = min(n, 5)
max_m = min(m, 5)

print
print ' Printing at most 5 first components of vectors'
print

print 'Initial point: ', x0[:max_n]
print 'Lower bounds on x: ', nlp.Lvar[:max_n]
print 'Upper bounds on x: ', nlp.Uvar[:max_n]
print 'f(x0) = ', nlp.obj(x0)
g0 = nlp.grad(x0)
print 'grad f(x0) = ', g0[:max_n]

if max_m > 0:
    print 'Initial multipliers: ', pi0[:max_m]
    print 'Lower constraint bounds: ', nlp.Lcon[:max_m]
    print 'Upper constraint bounds: ', nlp.Ucon[:max_m]
    c0 = nlp.cons(x0)
    print 'c(x0) = ', c0[:max_m]

J = nlp.jac(x0)
H = nlp.hess(x0, pi0)
print
print ' nnzJ = ', J.nnz
print ' nnzH = ', H.nnz

print
print ' Printing at most first 5x5 principal submatrix'
print

print 'J(x0) = ', J[:max_m, :max_n]
print 'Hessian (lower triangle):', H[:max_n, :max_n]

print
print ' Evaluating constraints individually, sparse gradients'
print

for i in range(max_m):
    ci = nlp.icons(i, x0)
    print 'c%d(x0) = %-g' % (i, ci)
    sgi = nlp.sigrad(i, x0)
    k = sgi.keys()
    ssgi = {}
    for j in range(min(5, len(k))):
        ssgi[k[j]] = sgi[k[j]]
    print 'grad c%d(x0) = ' % i, ssgi

print
print ' Testing matrix-vector product:'
print

e = numpy.ones(n, 'd')
e[0] = 2
e[1] = -1
He = nlp.hprod(x0, pi0, e)
print 'He = ', He[:max_n]


print
print ' Testing objective scaling:'
print

g = nlp.grad(x0)
print 'Maximum/Minimum gradient (unscaled): %12.5e / %12.5e' \
      % (max(abs(g)), min(abs(g)))
nlp.compute_scaling_obj()  # default is to use x0
g = nlp.grad(x0)
print 'Maximum/Minimum gradient ( scaled): %12.5e / %12.5e' \
      % (max(abs(g)), min(abs(g)))
nlp.compute_scaling_obj(reset=True)
g = nlp.grad(x0)
print '... and after a ''reset'' ...'
print 'Maximum/Minimum gradient (unscaled): %12.5e / %12.5e' \
      % (max(abs(g)), min(abs(g)))

print
print ' Testing constraint scaling:'
print

for i in xrange(max_m):
    nlp.compute_scaling_cons(reset=True)
    sgi = nlp.sigrad(i, x0)
    imax = max(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    imin = min(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    print 'Constraint %3i: Max/Min gradient (unscaled): %12.5e (%3i) / %12.5e (%3i)' \
          % (i, sgi[imax], imax, sgi[imin], imin)
    nlp.compute_scaling_cons()
    sgi = nlp.sigrad(i, x0)
    imax = max(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    imin = min(sgi.values, key=lambda x: abs(sgi.values.get(x)))
    print 'Constraint %3i: Max/Min gradient ( scaled): %12.5e (%3i) / %12.5e (%3i)' \
          % (i, sgi[imax], imax, sgi[imin], imin)

# Output "solution"
nlp.writesol(x0, pi0, 'And the winner is')
