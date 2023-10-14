#!/usr/bin/env bash

echo "Running init code"

if ! cd ~/mpi/code/
then
    echo "Changing directory failed"
    exit 1
fi    

if ! nr_threads=$(grep -c "processor" /proc/cpuinfo)
then
    echo "Getting the number of threads failed"
    exit 2
fi

if ! PYTHON=python3 ./bootstrap.sh -j "$nr_threads"
then
    echo "Building failed"
    exit 3
fi
