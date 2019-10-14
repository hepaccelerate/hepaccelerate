#!/bin/bash

set -e

function run() {
    export SINGULARITY_IMAGE=/storage/user/jpata/gpuservers/singularity/images/cupy.simg
    export NUMBA_NUM_THREADS=$1
    export OMP_NUM_THREADS=$1
    export HEPACCELERATE_CUDA=$2
    export PYTHONPATH=.
    export CUDA_VISIBLE_DEVICES=0
    export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
    export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/
    singularity exec --nv $SINGULARITY_IMAGE python3 tests/kernel_test.py
}

run 1 0
run 2 0
run 4 0
run 8 0
run 18 0
run 24 0
run 1 1
