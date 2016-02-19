# Build instructions for Mac OS/X users

Installing NLP.py will be much easier if you use [Homebrew](https://brew.sh).
Follow the instructions to install Homebrew.
Then, the following dependencies can be installed automatically:

```bash
brew install gcc  # currently v5. Contains gfortran

brew tap homebrew/science
brew install adol-c                             # will also install Colpack
brew install boost --with-mpi --without-single  # to use pycppad
brew install cppad --with-adol-c --with-boost --cc=gcc-4.9
brew install asl
brew install metis

pip install algopy
pip install git+https://github.com/b45ch1/pycppad.git
```

## Installing PyAdolc

```bash
git clone https://github.com/b45ch1/pyadolc.git
cd pyadolc
BOOST_DIR=$(brew --prefix boost-python) ADOLC_DIR=$(brew --prefix adol-c) COLPACK_DIR=$(brew --prefix colpack) python setup.py install
cd
python -c "import adolc; adolc.test()"
```
