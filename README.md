# NLP.py

[![Build Status](https://travis-ci.com/PythonOptimizers/NLP.py.svg?token=MZezWHtArpsrWZ3Yyqzx&branch=develop)](https://travis-ci.com/PythonOptimizers/NLP.py)
[![Coverage Status](https://coveralls.io/repos/github/PythonOptimizers/NLP.py/badge.svg?branch=develop)](https://coveralls.io/github/PythonOptimizers/NLP.py?branch=develop)

`NLP.py` is a Python package for modeling and solving continuous optimization problems.

## Dependencies

- [`Numpy`](http://www.numpy.org)
- [`PyKrylov`](https://github.com/PythonOptimizers/pykrylov)

## Optional dependencies

### Sparse matrix storage

One of:

- [`CySparse`](https://github.com/PythonOptimizers/cysparse) (recommended)
- [`PySparse`](https://github.com/optimizers/pysparse.git)
- [`Scipy`](http://scipy.org/scipylib) (does not have full support yet)

Only certain numerical methods and functionalities are available without sparse matrix support.

### Derivatives computation

At least one of the following, depending on requirements:

- [`ASL`](https://github.com/ampl/mp)
- [`pyadolc`](https://github.com/b45ch1/pyadolc.git)
- [`algopy`](https://github.com/b45ch1/algopy.git)
- [`pycppad`](https://github.com/b45ch1/pycppad.git)

Without one of the above dependencies, at least the first derivatives must be coded by hand. Second derivatives may be approximated used a quasi-Newton scheme.

### Factorizations

One or more of the following, depending on requirements:

- [`HSL.py`](https://github.com/PythonOptimizers/HSL.py)
- [`MUMPS.py`](https://github.com/PythonOptimizers/MUMPS.py)
- [`qr_mumps.py`](https://github.com/PythonOptimizers/qr_mumps.py)
- [`SuiteSparse.py`](https://github.com/PythonOptimizers/SuiteSparse.py)

Only matrix-free methods are available without one of the above factorizations.

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/PythonOptimizers/NLP.py
   ```

2. Optional: Install optional dependencies. OSX users, see `Readme.osx`.

3. Optional: if you would like ASL support, copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration::
    ```bash
    cp site.template.cfg site.cfg
    ```

4. Install:
    ```bash
    python setup.py build
    python setup.py install [--prefix=...]
    ```
