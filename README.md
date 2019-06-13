[![Build Status](https://travis-ci.com/jpata/hepaccelerate.svg?branch=master)](https://travis-ci.com/jpata/hepaccelerate)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate/badges/libhmm/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate/commits/libhmm)

# hepaccelerate

Accelerated array analysis on flat ROOT data. Process 1 billion events to histograms in minutes on a single workstation.
Weighted histograms, jet-lepton deltaR matching and more! Works on both the CPU and GPU!

## Installation

~~~
pip install git+https://github.com/jpata/hepaccelerate@v0.1.0
~~~

Required python libraries:
 - python 3
 - uproot
 - awkward-array
 - numba (>0.43)

Optional libraries for CUDA acceleration:
 - cupy
 - cudatoolkit

## Usage

This is a minimal example from [tests/example.py](tests/example.py).

```python
#!/usr/bin/env python3
import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

#choose whether or not to use the GPU backend
NUMPY_LIB, ha = choose_backend(use_cuda=False)

#define our analysis function
def analyze_data_function(data, parameters):
    ret = Results()

    num_events = data["num_events"]
    muons = data["Muon"]
    mu_pt = NUMPY_LIB.sqrt(muons.Px**2 + muons.Py**2)
    muons.attrs_data["pt"] = mu_pt

    mask_events = NUMPY_LIB.ones(muons.numevents(), dtype=NUMPY_LIB.bool)
    mask_muons_passing_pt = muons.pt > parameters["muons_ptcut"]
    num_muons_event = ha.sum_in_offsets(muons, mask_muons_passing_pt, mask_events, muons.masks["all"], NUMPY_LIB.int8)
    mask_events_dimuon = num_muons_event == 2

    #get the leading muon pt in events that have exactly two muons
    inds = NUMPY_LIB.zeros(num_events, dtype=NUMPY_LIB.int32)
    leading_muon_pt = ha.get_in_offsets(muons.pt, muons.offsets, inds, mask_events_dimuon, mask_muons_passing_pt)

    #compute a weighted histogram
    weights = NUMPY_LIB.ones(num_events, dtype=NUMPY_LIB.float32)
    bins = NUMPY_LIB.linspace(0,300,101)
    hist_muons_pt = Histogram(*ha.histogram_from_vector(leading_muon_pt[mask_events_dimuon], weights[mask_events_dimuon], bins))

    #save it to the output
    ret["hist_leading_muon_pt"] = hist_muons_pt
    return ret

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

dataset = Dataset("HZZ", [filename], datastructures, cache_location="./mycache/", treename="events", datapath="")

#load data to memory
try:
    dataset.from_cache()
    print("Loaded data from cache, did not touch original ROOT files.")
except FileNotFoundError as e:
    print("Cache not found, creating...")
    dataset.load_root()
    dataset.to_cache()

#process data
results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
```
