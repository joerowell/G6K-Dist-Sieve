# Building for MPI

This document contains details on how to build G6K to use MPI.

## Requirements
- An MPI installation. If you're using conda, you can install OpenMPI using ` conda install -c conda-forge openmpi`.
- [Doctest](https://github.com/doctest/doctest). You need to clone doctest to the `code` directory (i.e the current directory).


## Building

### Building for Cython
We support autotools for building this code. The only tricky thing is that you need to explicitly tell autotools to use the MPI wrapper around the C++ and C compilers. 
This is  primarily for historical reasons, but MPI still requires this. We also make it so that you need to pass in a flag to enable MPI, just in case you don't need it.

Thus, building the C++ code looks like this:

```shell
CC=mpicc CXX=mpicxx ./configure --enable-mpi
make
```



### Building tests
TODO

## Limitations
Please note that if you are using this code, you must set the distribution requirements such that
at most 2^31 vectors are sent at once. This means that the distribution threshold needs to be suitably low. This is explained in more detail in ```mpi_wrapper.hpp```. 
