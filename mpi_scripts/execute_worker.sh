#!/usr/bin/env bash

# This script passes the threads argument that was passed in.
if ! TEMP=$(getopt -o t: --long threads: -- "$@")
then
    echo "No threads were passed on $(hostname)" &>2
    exit 1
fi

# Evaluate the arguments.
eval set -- "$TEMP"

threads=0
while true; do
    case "$1" in
        -t | --threads ) threads="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

# Check that threads was set.

if [ "$threads" -eq 0 ]; then
    echo "Threads argument was set to 0 on $(hostname)";
    exit 2
fi    

cd ~/mpi/code/ || echo "Failed to navigate to ~mpi/code on $(hostname)" && exit
./kernel/dist_siever --threads "$threads"
