#!/bin/bash

dask-scheduler &

sleep 10

for idev in `seq 0 7`; do
    for iprl in `seq 0 1`; do
        CUDA_VISIBLE_DEVICES=$idev PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &
    done
done

wait
