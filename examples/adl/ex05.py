#usr/bin/env python3
#Run as PYTHONPATH=. python3 examples/adl/ex05.py
#Plot the missing ET of events that have an opposite-sign muon pair with an invariant mass between 60 and 120 GeV.

import math
import numpy as np

import numba
import numba.cuda as cuda
import hepaccelerate.backend_cuda as backend_cuda
import hepaccelerate.backend_cpu as backend_cpu

import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Histogram
from boilerplate import run

#define our analysis function
def analyze_data_function(data, parameters):
    backend = parameters["backend"]
    nplib = parameters["nplib"]

    ret = Results()

    num_events = data["num_events"]
    mask_events = nplib.ones(num_events, dtype=nplib.bool)

    muons = data["Muon"]
    mask_all_muons = nplib.ones(muons.numobjects(), nplib.bool)

    event_mask_out = np.zeros(num_events, dtype=nplib.bool)
    backend.select_events(muons.charge, muons.pt, muons.eta, muons.phi, muons.mass, muons.offsets, mask_all_muons, event_mask_out)
    print(nplib.sum(event_mask_out))
 
    return ret

datastructures = {
   "Muon": [
       ("Muon_pt", "float32"),
       ("Muon_eta", "float32"),
       ("Muon_phi", "float32"),
       ("Muon_mass", "float32"),
       ("Muon_charge", "int32"),
    ],
   "EventVariables": [
       ("MET_sumet", "float32"),
   ]
}

from hepaccelerate.backend_cpu import spherical_to_cartesian
   
@numba.njit(parallel=True)
def select_events_cpu(charge, pt, eta, phi, mass, offsets, content_mask_in, event_mask_out):
    for iev in numba.prange(offsets.shape[0]-1):
        start = np.uint64(offsets[iev])
        end = np.uint64(offsets[iev + 1])
        
        for iobj in range(start, end):
            cm1 = content_mask_in[iobj]
            ch1 = charge[iobj]

            for jobj in range(iobj + 1, end):
                cm2 = content_mask_in[jobj]
                ch2 = charge[jobj]

                px1, py1, pz1, e1 = spherical_to_cartesian(pt[iobj], eta[iobj], phi[iobj], mass[iobj])
                px2, py2, pz2, e2 = spherical_to_cartesian(pt[jobj], eta[jobj], phi[jobj], mass[jobj])

                inv_mass = np.sqrt(-((px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2 - (e1+e2)**2))
                inv_mass_window = (inv_mass > 60.0) & (inv_mass < 120.0)
                event_mask_out[iev] = (cm1&cm2) & (ch1!=ch2) & inv_mass_window
    return

from hepaccelerate.backend_cuda import spherical_to_cartesian_devfunc

@cuda.jit
def select_events_cuda_kernel(charge, pt, eta, phi, mass, offsets, content_mask_in, event_mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        start = np.uint64(offsets[iev])
        end = np.uint64(offsets[iev + 1])
        
        for iobj in range(start, end):
            cm1 = content_mask_in[iobj]
            ch1 = charge[iobj]

            px1, py1, pz1, e1 = spherical_to_cartesian_devfunc(pt[iobj], eta[iobj], phi[iobj], mass[iobj])

            for jobj in range(iobj + 1, end):
                cm2 = content_mask_in[jobj]
                ch2 = charge[jobj]

                px2, py2, pz2, e2 = spherical_to_cartesian_devfunc(pt[jobj], eta[jobj], phi[jobj], mass[jobj])

                inv_mass = math.sqrt(-((px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2 - (e1+e2)**2))
                inv_mass_window = (inv_mass > 60.0) & (inv_mass < 120.0)
                event_mask_out[iev] = (cm1&cm2) & (ch1!=ch2) & inv_mass_window
    return

def select_events_cuda(charge, pt, eta, phi, mass, offsets, content_mask_in, event_mask_out):
    select_events_cuda_kernel[32, 1024](charge, pt, eta, phi, mass, offsets, content_mask_in, event_mask_out)

backend_cuda.select_events = select_events_cuda
backend_cpu.select_events = select_events_cpu

if __name__ == "__main__":
    run(analyze_data_function, datastructures, "out_ex01.json")
