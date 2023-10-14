#!/usr/bin/env bash

# This script exists to get around an annoying issue with MPICH's exec files: the inability
# to pass arguments directly to the underlying program.
# To use: pass the arguments as you would for running e.g svp_challenge directly.
TEMP=$(getopt -o d:t:s:h:b:v --long dimension:,threads:,sieve:,dist_threshold:,bucket_batches:,scale_factor: -- "$@")

if ! TEMP=$(getopt -o d:t:s:h:b:v --long dimension:,threads:,sieve:,dist_threshold:,bucket_batches:,scale_factor: -- "$@")
then
    echo "Running getopt failed..." >&2 
    exit 1
fi

name=$(hostname)

if [ "$name" != "holzfusion" ]; then
    echo "This script expects to be run from Holzfusion."
    exit 2
fi

# Evaluate the arguments.
eval set -- "$TEMP"

# This assumes that the program will be run from Holzfusion.
threads=48
sieve=
dist_threshold=
bucket_batches=1
scale_factor=1
dimension=

while true; do
    case "$1" in
        -d | --dimension ) dimension="$2"; shift 2 ;;
        -t | --threads ) threads="$2"; shift 2 ;;
        -s | --sieve ) sieve="$2"; shift 2 ;;
        -h | --dist_threshold) dist_threshold="$2"; shift 2 ;;
        -b | --bucket_batches) bucket_batches="$2"; shift 2 ;;
        -v | --scale_factor) scale_factor="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

# Check arguments.

if [ ${#sieve} -eq 0 ]; then
    echo "No sieve specified."
    echo "$sieve"
    exit 3
fi

if [ ${#dimension} -eq 0 ]; then
    echo "No dimension specified."
    exit 4
fi

if [ ${#dist_threshold} -eq 0 ]; then
    echo "No dist_threshold specified."
    exit 5
fi

# Now we need to set them as environment variables for mpich's consumption
mpiexec -verbose -env MP_DIMENSION "$dimension" -env MP_THREADS "$threads" -env MP_SIEVE "$sieve" -env MP_DT "$dist_threshold" -env MP_BB "$bucket_batches" -env MP_SF "$scale_factor" -f hosts -configfile run_svp_challenge_config
