#!/bin/bash

dask-scheduler &

sleep 10

for i in `seq 1 $1`; do 
    PYTHONPATH=.:examples dask-worker 127.0.0.1:8786 --no-nanny --nthreads 1 --nprocs 1 --memory-limit 0 &
done

wait
