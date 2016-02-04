NLPy2.0
=======
New version of NLPy with satellites


Dependencies
------------

- Numpy
- [`PyKrylov`](https://github.com/PythonOptimizers/pykrylov)


Optional dependencies
---------------------

A strongly recommended dependency is the AMPL modeling language Solver Library

Python dependencies:

- [`CySparse`](https://github.com/PythonOptimizers/cysparse) (the successor of PySparse)
- [`PySparse`](https://github.com/optimizers/pysparse.git)
- [`Scipy`](http://scipy.org/scipylib/)
- [`pyadolc`](https://github.com/b45ch1/pyadolc.git) 
- [`algopy`](https://github.com/b45ch1/algopy.git)
- [`pycppad`](https://github.com/b45ch1/pycppad.git)
- [`HSL.py`](https://github.com/PythonOptimizers/HSL.py)
- [`MUMPS.py`](https://github.com/PythonOptimizers/MUMPS.py`)
- [`qr_mumps.py`](https://github.com/PythonOptimizers/qr_mumps.py)
- [`SuiteSparse.py`](https://github.com/PythonOptimizers/SuiteSparse.py)



Installation
------------

1. Clone this repo::

        git clone https://github.com/PythonOptimizers/NLPy2.0


2. Copy `site.template.cfg` to `site.cfg` and modify `site.cfg` to match your configuration::
        
        cp site.template.cfg site.cfg
    
    
3. Install::

        python setup.py build
        python setup.py install [--prefix=...]
