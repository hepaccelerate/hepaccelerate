#!/bin/bash
set -e

#compiled from singularity/cupy-tf-gpu.singularity
SIMG=~/gpuservers/singularity/images/cupy.simg

#download the files using `examples/download_example_data.sh ./cms_opendata_files`
DATAPATH=./cms_opendata_files

NTHREADS=24

#Start the dask cluster (multi-CPU)
singularity exec $SIMG ./examples/dask_cluster.sh $NTHREADS &

#Run the analysis (few systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=0 singularity exec --nv $SIMG python3 examples/full_analysis.py --out data/out_cpu_nt24_njec1.pkl --datapata $DATAPATH  > data/log_cpu_njec1.txt

#Start the dask cluster (multi-CPU)
singularity exec $SIMG ./examples/dask_cluster.sh $NTRHEADS &

#Run the analysis (many systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=0 singularity exec --nv $SIMG python3 examples/full_analysis.py --njec 20 --out data/out_cpu_nt24_njec20.pkl --datapath $DATAPATH > data/log_cpu_njec20.txt

#Start the dask cluster (multi-GPU)
singularity exec --nv $SIMG ./examples/dask_cluster_gpu.sh &

#Run the analysis (few systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=1 singularity exec --nv $SIMG python3 examples/full_analysis.py --out data/out_gpu_nt4_njec1.pkl --datapath $DATAPATH > data/log_gpu_njec1.txt

#Start the dask cluster (multi-GPU)
singularity exec --nv $SIMG ./examples/dask_cluster_gpu.sh &

#Run the analysis (many systematics)
PYTHONPATH=. HEPACCELERATE_CUDA=1 singularity exec --nv $SIMG python3 examples/full_analysis.py --njec 20 --out data/out_gpu_nt4_njec20.pkl --datapath $DATAPATH > data/log_gpu_njec20.txt
