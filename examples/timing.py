import requests
import os
import numpy as np
import json
import sys
import time
import uproot
import numba

import hepaccelerate
import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
from tests.kernel_test import load_dataset
 
USE_CUDA = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
nplib, backend = choose_backend(use_cuda=USE_CUDA)

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
    sel_ev = nplib.ones(muons.numevents(), dtype=nplib.bool)
    sel_mu = nplib.ones(muons.numobjects(), dtype=nplib.bool)
    z = kernels.sum_in_offsets(
        backend,
        muons.offsets,
        muons.pt,
        sel_ev,
        sel_mu, dtype=nplib.float32)

def test_kernel_simple_cut(dataset):
    muons = dataset.structs["Muon"][0]
    sel_mu = muons.pt > 30.0

def test_kernel_max_in_offsets(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = nplib.ones(muons.numevents(), dtype=nplib.bool)
    sel_mu = nplib.ones(muons.numobjects(), dtype=nplib.bool)
    z = kernels.max_in_offsets(
        backend,
        muons.offsets,
        muons.pt,
        sel_ev,
        sel_mu)
    
def test_kernel_get_in_offsets(dataset):
   muons = dataset.structs["Muon"][0]
   sel_ev = nplib.ones(muons.numevents(), dtype=nplib.bool)
   sel_mu = nplib.ones(muons.numobjects(), dtype=nplib.bool)
   inds = nplib.zeros(muons.numevents(), dtype=nplib.int8)
   inds[:] = 0
   z = kernels.get_in_offsets(
       backend,
       muons.offsets,
       muons.pt,
       inds,
       sel_ev,
       sel_mu)

def test_kernel_mask_deltar_first(dataset):
    muons = dataset.structs["Muon"][0]
    jet = dataset.structs["Jet"][0]
    sel_ev = nplib.ones(muons.numevents(), dtype=nplib.bool)
    sel_mu = nplib.ones(muons.numobjects(), dtype=nplib.bool)
    sel_jet = (jet.pt > 10)
    muons_matched_to_jet = kernels.mask_deltar_first(
        backend,
        {"offsets": muons.offsets, "eta": muons.eta, "phi": muons.phi},
        sel_mu,
        {"offsets": jet.offsets, "eta": jet.eta, "phi": jet.phi},
        sel_jet, 0.3
    )

def test_kernel_histogram_from_vector(dataset):
    muons = dataset.structs["Muon"][0]
    weights = 2*nplib.ones(muons.numobjects(), dtype=nplib.float32)
    ret = kernels.histogram_from_vector(backend, muons.pt, weights, nplib.linspace(0,200,100, dtype=nplib.float32))

def test_kernel_histogram_from_vector_several(dataset):
    muons = dataset.structs["Muon"][0]
    mask = nplib.ones(muons.numobjects(), dtype=nplib.bool)
    mask[:100] = False
    weights = 2*nplib.ones(muons.numobjects(), dtype=nplib.float32)
    variables = [
        (muons.pt, nplib.linspace(0,200,100, dtype=nplib.float32)),
        (muons.eta, nplib.linspace(-4,4,100, dtype=nplib.float32)),
        (muons.phi, nplib.linspace(-4,4,100, dtype=nplib.float32)),
        (muons.mass, nplib.linspace(0,200,100, dtype=nplib.float32)),
        (muons.charge, nplib.array([-1, 0, 1, 2], dtype=nplib.float32)),
    ]
    ret = kernels.histogram_from_vector_several(backend, variables, weights, mask)
    
def test_kernel_select_opposite_sign(dataset):
    muons = dataset.structs["Muon"][0]
    sel_ev = nplib.ones(muons.numevents(), dtype=nplib.bool)
    sel_mu = nplib.ones(muons.numobjects(), dtype=nplib.bool)
    muons_passing_os = kernels.select_opposite_sign(
        backend,
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
        ds.move_to_device(nplib)
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
    dataset = load_dataset(nplib, 5)
    test_timing(dataset)
