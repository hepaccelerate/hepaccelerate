#usr/bin/env python3
#Run as PYTHONPATH=. python3 examples/simple_hzz.py

#In case you use CUDA, you may have to find the libnvvm.so on your system manually
import os
import numba
import sys
import numpy as np

import hepaccelerate
import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#define our analysis function
def analyze_data_function(data, parameters):
    ret = Results()

    num_events = data["num_events"]
    muons = data["Muon"]
    mu_pt = nplib.sqrt(muons.Px**2 + muons.Py**2)
    muons.attrs_data["pt"] = mu_pt

    mask_events = nplib.ones(muons.numevents(), dtype=nplib.bool)
    mask_muons_passing_pt = muons.pt > parameters["muons_ptcut"]
    num_muons_event = kernels.sum_in_offsets(backend, muons.offsets, mask_muons_passing_pt, mask_events, muons.masks["all"], nplib.int8)
    mask_events_dimuon = num_muons_event == 2

    #get the leading muon pt in events that have exactly two muons
    inds = nplib.zeros(num_events, dtype=nplib.int32)
    leading_muon_pt = kernels.get_in_offsets(backend, muons.offsets, muons.pt, inds, mask_events_dimuon, mask_muons_passing_pt)

    #compute a weighted histogram
    weights = nplib.ones(num_events, dtype=nplib.float32)
    bins = nplib.linspace(0,300,101, dtype=nplib.float32)
    hist_muons_pt = Histogram(*kernels.histogram_from_vector(backend, leading_muon_pt[mask_events_dimuon], weights[mask_events_dimuon], bins))

    #save it to the output
    ret["hist_leading_muon_pt"] = hist_muons_pt
    return ret

if __name__ == "__main__":
    #choose whether or not to use the GPU backend
    use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
    if use_cuda:
        import setGPU
    
    nplib, backend = choose_backend(use_cuda=use_cuda)
  
    #Load this input file
    filename = "data/HZZ.root"
    
    #Predefine which branches to read from the TTree and how they are grouped to objects
    #This will be verified against the actual ROOT TTree when it is loaded
    datastructures = {
                "Muon": [
                    ("Muon_Px", "float32"),
                    ("Muon_Py", "float32"),
                    ("Muon_Pz", "float32"), 
                    ("Muon_E", "float32"),
                    ("Muon_Charge", "int32"),
                    ("Muon_Iso", "float32")
                ],
                "Jet": [
                    ("Jet_Px", "float32"),
                    ("Jet_Py", "float32"),
                    ("Jet_Pz", "float32"),
                    ("Jet_E", "float32"),
                    ("Jet_btag", "float32"),
                    ("Jet_ID", "bool")
                ],
                "EventVariables": [
                    ("NPrimaryVertices", "int32"),
                    ("triggerIsoMu24", "bool"),
                    ("EventWeight", "float32")
                ]
        }
   
    #Define a dataset, given the data structure and a list of filenames 
    dataset = Dataset("HZZ", [filename], datastructures, treename="events")
   
    #Load the ROOT files 
    dataset.load_root(verbose=True)
    
    #merge arrays across files into one big array
    dataset.merge_inplace(verbose=True)
    
    #move to GPU if CUDA was specified
    dataset.move_to_device(nplib, verbose=True)
    
    #process data, save output as a json file
    results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
    results.save_json("out.json")

    #Make a simple PDF plot as an example
    hist = results["hist_leading_muon_pt"]
    fig = plt.figure(figsize=(5,5))
    plt.errorbar(hist.edges[:-1], hist.contents, np.sqrt(hist.contents_w2))
    plt.savefig("hist.png", bbox_inches="tight")
