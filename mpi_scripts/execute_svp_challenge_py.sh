#!/usr/bin/env bash

# Preserve, just in case we lose them due to switching environment (see below). 
threads=$MP_THREADS
dimension=$MP_DIMENSION
sieve=$MP_SIEVE
dist_threshold=$MP_DT
bucket_batches=$MP_SF
scale_factor=$MP_SF

cd ~/mpi/code/ || echo "Failed to navigate to ~/mpi/code on $(hostname)" && exit

if ! test -f "activate"; then
    echo "activate file does not on $(hostname)"
    exit 1
fi

# This is covered by the above
# shellcheck source=/dev/null
source ./activate || echo "Failed to activate on $(hostname)" && exit

python3 ./svp_challenge.py "$dimension" --threads "$threads" --dist_threshold "$dist_threshold" --sieve "$sieve" --bucket_batches "$bucket_batches" --scale_factor "$scale_factor"

retval=$?

if [ $retval -ne 0 ]; then
    echo "Error sieving"
fi

## This will always pass.
# shellcheck source=/dev/null
source deactivate

exit $retval
