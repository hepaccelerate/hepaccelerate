#!/bin/bash
set -e

#Start the dask cluster (multi-CPU)
singularity exec ~/gpuservers/singularity/images/cupy.simg ./examples/dask_cluster.sh 24 &

#Run the analysis (few systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=0 singularity exec --nv -B /storage ~/gpuservers/singularity/images/cupy-v2.simg python3 examples/full_analysis.py --out data/out_cpu_nt24_njec1.txt --datapath /storage/user/jpata > log_cpu_njec1.txt

#Start the dask cluster (multi-CPU)
singularity exec ~/gpuservers/singularity/images/cupy.simg ./examples/dask_cluster.sh 24 &

#Run the analysis (many systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=0 singularity exec --nv -B /storage ~/gpuservers/singularity/images/cupy-v2.simg python3 examples/full_analysis.py --njec 20 --out data/out_cpu_nt24_njec20.txt --datapath /storage/user/jpata > log_cpu_njec20.txt

#Start the dask cluster (multi-GPU)
singularity exec --nv ~/gpuservers/singularity/images/cupy-tf-gpu.simg ./examples/dask_cluster_gpu.sh &

#Run the analysis (few systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=1 singularity exec --nv -B /storage ~/gpuservers/singularity/images/cupy-tf-gpu-v2.simg python3 examples/full_analysis.py --out data/out_gpu_nt4_njec1.txt --datapath /storage/user/jpata > log_gpu_njec1.txt

#Start the dask cluster (multi-GPU)
singularity exec --nv ~/gpuservers/singularity/images/cupy-tf-gpu.simg ./examples/dask_cluster_gpu.sh &

#Run the analysis (many systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=1 singularity exec --nv -B /storage ~/gpuservers/singularity/images/cupy-tf-gpu-v2.simg python3 examples/full_analysis.py --njec 20 --out data/out_gpu_nt4_njec20.txt --datapath /storage/user/jpata > log_gpu_njec20.txt
