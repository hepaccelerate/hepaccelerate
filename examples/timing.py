import requests
import os
import numpy as np
import json
import sys
import time
import uproot
import numba

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
from tests.kernel_test import load_dataset
 
USE_CUDA = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
NUMPY_LIB, ha = choose_backend(use_cuda=USE_CUDA)

def time_kernel(dataset, test_kernel):
    #ensure it's compiled
    test_kernel(dataset)

    n = len(dataset)

    t0 = time.time()
    for i in range(5):
        test_kernel(dataset)
    t1 = time.time()

    dt = (t1 - t0) / 5.0
    speed = float(n)/dt
    return speed

def test_kernel_sum_in_offsets(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    z = ha.sum_in_offsets(
        muons.offsets,
        muons.pt,
        sel_ev,
        sel_mu, dtype=NUMPY_LIB.float32)

def test_kernel_simple_cut(dataset):
    muons = dataset.structs["Muon"][0]
    sel_mu = muons.pt > 30.0

def test_kernel_max_in_offsets(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    z = ha.max_in_offsets(
        muons.offsets,
        muons.pt,
        sel_ev,
        sel_mu)
    
def test_kernel_get_in_offsets(dataset):
   muons = dataset.structs["Muon"][0]
   sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
   sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
   inds = NUMPY_LIB.zeros(muons.numevents(), dtype=NUMPY_LIB.int8)
   inds[:] = 0
   z = ha.get_in_offsets(
       muons.offsets,
       muons.pt,
       inds,
       sel_ev,
       sel_mu)

def test_kernel_mask_deltar_first(dataset):
    muons = dataset.structs["Muon"][0]
    jet = dataset.structs["Jet"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    sel_jet = (jet.pt > 10)
    muons_matched_to_jet = ha.mask_deltar_first(
        {"offsets": muons.offsets, "eta": muons.eta, "phi": muons.phi},
        sel_mu,
        {"offsets": jet.offsets, "eta": jet.eta, "phi": jet.phi},
        sel_jet, 0.3
    )

def test_kernel_histogram_from_vector(dataset):
    muons = dataset.structs["Muon"][0]
    weights = 2*NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.float32)
    ret = ha.histogram_from_vector(muons.pt, weights, NUMPY_LIB.linspace(0,200,100, dtype=NUMPY_LIB.float32))

def test_kernel_histogram_from_vector_several(dataset):
    muons = dataset.structs["Muon"][0]
    mask = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    mask[:100] = False
    weights = 2*NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.float32)
    variables = [
        (muons.pt, NUMPY_LIB.linspace(0,200,100, dtype=NUMPY_LIB.float32)),
        (muons.eta, NUMPY_LIB.linspace(-4,4,100, dtype=NUMPY_LIB.float32)),
        (muons.phi, NUMPY_LIB.linspace(-4,4,100, dtype=NUMPY_LIB.float32)),
        (muons.mass, NUMPY_LIB.linspace(0,200,100, dtype=NUMPY_LIB.float32)),
        (muons.charge, NUMPY_LIB.array([-1, 0, 1, 2], dtype=NUMPY_LIB.float32)),
    ]
    ret = ha.histogram_from_vector_several(variables, weights, mask)
    
def test_kernel_select_opposite_sign(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    sel_mu = NUMPY_LIB.ones(muons.numobjects(), dtype=NUMPY_LIB.bool)
    muons_passing_os = ha.select_opposite_sign(
        muons.offsets, muons.charge, sel_mu)

def test_timing(ds):
    with open("data/kernel_benchmarks.txt", "a") as of:
        for i in range(5):
            ret = run_timing(ds)
            of.write(json.dumps(ret) + '\n')

def run_timing(ds):
    print("Testing memory transfer speed")
    t0 = time.time()
    for i in range(5):
        ds.move_to_device(NUMPY_LIB)
    t1 = time.time()
    dt = (t1 - t0)/5.0

    ret = {
        "use_cuda": USE_CUDA, "num_threads": numba.config.NUMBA_NUM_THREADS,
        "use_avx": numba.config.ENABLE_AVX, "num_events": ds.numevents(),
        "memsize": ds.memsize()
    }

    print("Memory transfer speed: {0:.2f} MHz, event size {1:.2f} bytes, data transfer speed {2:.2f} MB/s".format(
        ds.numevents() / dt / 1000.0 / 1000.0, ds.eventsize(), ds.memsize()/dt/1000/1000))
    ret["memory_transfer"] = ds.numevents() / dt / 1000.0 / 1000.0

    t = time_kernel(ds, test_kernel_sum_in_offsets)
    print("sum_in_offsets {0:.2f} MHz".format(t/1000/1000))
    ret["sum_in_offsets"] = t/1000/1000

    t = time_kernel(ds, test_kernel_simple_cut)
    print("simple_cut {0:.2f} MHz".format(t/1000/1000))
    ret["simple_cut"] = t/1000/1000

    t = time_kernel(ds, test_kernel_max_in_offsets)
    print("max_in_offsets {0:.2f} MHz".format(t/1000/1000))
    ret["max_in_offsets"] = t/1000/1000

    t = time_kernel(ds, test_kernel_get_in_offsets)
    print("get_in_offsets {0:.2f} MHz".format(t/1000/1000))
    ret["get_in_offsets"] = t/1000/1000

    t = time_kernel(ds, test_kernel_mask_deltar_first)
    print("mask_deltar_first {0:.2f} MHz".format(t/1000/1000))
    ret["mask_deltar_first"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_select_opposite_sign)
    print("select_muons_opposite_sign {0:.2f} MHz".format(t/1000/1000))
    ret["select_muons_opposite_sign"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_histogram_from_vector)
    print("histogram_from_vector {0:.2f} MHz".format(t/1000/1000))
    ret["histogram_from_vector"] = t/1000/1000
    
    t = time_kernel(ds, test_kernel_histogram_from_vector_several)
    print("histogram_from_vector_several {0:.2f} MHz".format(t/1000/1000))
    ret["histogram_from_vector_several"] = t/1000/1000
    return ret

if __name__ == "__main__":
    dataset = load_dataset(NUMPY_LIB, 5)
    test_timing(dataset)
