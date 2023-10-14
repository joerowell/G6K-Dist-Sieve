#!/usr/bin/env bash

echo "Running rebuild code"

if ! cd ~/mpi/code/
then 
    echo "Changing directory failed"
    exit 1
fi    

if ! source ./activate
then
    echo "Activating failed"
    exit 2
fi



if ! CXX=mpicxx CC=mpicc ./configure --enable-mpi=yes --enable-mpidt=yes
then
    echo "Configuration failed"
    exit 2    
fi

if ! make clean
then
    echo "Making clean failed"
    exit 3
fi

if ! make -j
then
   echo "Making failed"
   exit 4
fi

if ! python3 setup.py build_ext --inplace
then
    echo "Making Python module failed"
    exit 5
fi
