#!/bin/bash

set -e

dosingularity=true

function run_kernels() {
    export NUMBA_NUM_THREADS=$1
    export NUMBA_USE_AVX=0
    export NUMBA_THREADING_LAYER=omp
    export OMP_NUM_THREADS=$1
    export MKL_NUM_THREADS=$1
    export HEPACCELERATE_CUDA=$2
    export PYTHONPATH=.
    export CUDA_VISIBLE_DEVICES=0
    if [ $dosingularity ]; then
        export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy.simg
        singularity exec --nv $SINGULARITY_IMAGE python3 examples/timing.py
    else
        python3 examples/timing.py
    fi
}

run_kernels 1 0
run_kernels 2 0
run_kernels 4 0
run_kernels 8 0
run_kernels 18 0
run_kernels 24 0
run_kernels 1 1
