#!/bin/bash
SINGULARITY_IMAGE=/bigdata/shared/Software/singularity/gpuservers/singularity/images/cupy.simg
singularity exec --nv $SINGULARITY_IMAGE python3 setup.py install --user
#NUMBA_NUM_THREADS=1 HEPACCELERATE_CUDA=0 singularity exec --nv $SINGULARITY_IMAGE  python3 tests/kernel_test.py
#NUMBA_NUM_THREADS=8 HEPACCELERATE_CUDA=0 singularity exec --nv $SINGULARITY_IMAGE  python3 tests/kernel_test.py
#NUMBA_NUM_THREADS=24 HEPACCELERATE_CUDA=0 singularity exec --nv $SINGULARITY_IMAGE  python3 tests/kernel_test.py
NUMBA_NUM_THREADS=24 HEPACCELERATE_CUDA=1 NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/ singularity exec --nv $SINGULARITY_IMAGE  python3 tests/kernel_test.py

