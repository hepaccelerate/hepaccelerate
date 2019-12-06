#!/bin/bash

dask-scheduler &

sleep 10

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &

wait
