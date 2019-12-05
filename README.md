[![Build Status](https://travis-ci.com/hepaccelerate/hepaccelerate.svg?branch=master)](https://travis-ci.com/hepaccelerate/hepaccelerate)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate/badges/master/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate/commits/master)
[![DOI](https://zenodo.org/badge/191644111.svg)](https://zenodo.org/badge/latestdoi/191644111)

# hepaccelerate

- HEP data analysis with [jagged arrays](https://github.com/scikit-hep/awkward-array) using python + [Numba](http://numba.pydata.org/)
- Use **any ntuple**, as long as you can open it with [uproot](https://github.com/scikit-hep/uproot)
- analyze a billion events with systematic to histograms in minutes on a single workstation
  - 1e9 events / (50 kHz x 24 threads) ~ 13 minutes
- weighted histograms, deltaR matching and [more](https://github.com/hepaccelerate/hepaccelerate#kernels)
- use a CPU or an nVidia CUDA GPU with the same interface!
- this is **not** an analysis framework, but rather a small set of helpers for fast jagged array processing

**Under active development and use by a few CMS analyses!**

<p float="left">
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/images/kernel_benchmarks.png" alt="Kernel benchmarks" width="300"/>
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/images/analysis_scaling.png" alt="Analysis scaling" width="300"/>
</p>

More details are available:
- writeup: https://arxiv.org/abs/1906.06242v2
- PyHEP 2019 talk: https://indico.cern.ch/event/833895/contributions/3577804/attachments/1927026/3192574/2019_10_15_pyhep.pdf

## Installation

~~~
pip install hepaccelerate
~~~

Required python libraries:
 - python 3
 - uproot
 - awkward-array
 - numba (>0.43)

Optional libraries:
 - cupy
 - cudatoolkit
 - dask

## Documentation
This code consists of two parts which can be used independently:
  - the accelerated HEP kernels that run on jagged data in [backend_cpu.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cpu.py) and [backend_cuda.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cuda.py)  
  - JaggedStruct, Dataset and Histogram classes to help with HEP dataset management

## Kernels

The kernels work on the basis of the `content` and `offsets` arrays based on `awkward.JaggedArray`
```python
import uproot, os
use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0))==1

if use_cuda:
  import hepaccelerate.backend_cuda as ha
  import cupy as np
  import numpy
else:
  import hepaccelerate.backend_cpu as ha
  import numpy as np
  np.asnumpy = np.array

fi = uproot.open("data/nanoaod_test.root")
tt = fi.get("aod2nanoaod/Events")
jet_eta = tt.array("Jet_eta")
jet_phi = tt.array("Jet_phi")
jet_pt = tt.array("Jet_pt")

lep_eta = tt.array("Muon_eta")
lep_phi = tt.array("Muon_phi")
lep_pt = tt.array("Muon_pt")

jets = {"pt": np.array(jet_pt.content), "eta": np.array(jet_eta.content), "phi": np.array(jet_phi.content), "offsets": np.array(jet_pt.offsets)}
sel_jets = jet_pt.content > 30.0

leptons = {"pt": np.array(lep_pt.content), "eta": np.array(lep_eta.content), "phi": np.array(lep_phi.content), "offsets": np.array(lep_pt.offsets)}
sel_leptons = lep_pt.content > 20.0

#run multithreaded CPU kernels
max_jet_pt = ha.max_in_offsets(jets["offsets"], jets["pt"])

#compare with awkward-array
m1 = np.asnumpy(max_jet_pt)
m2 = jet_pt.max()
m2[numpy.isinf(m2)] = 0
assert(numpy.all(m1 == m2))

masked_jets = ha.mask_deltar_first(jets, sel_jets, leptons, sel_leptons, 0.5)
print("kept {0} jets out of {1}".format(masked_jets.sum(), len(masked_jets)))
```

We have implemented the following kernels for both the CPU and CUDA backends:
  - `ha.min_in_offsets(offsets, content, mask_rows, mask_content)`: retrieve the minimum value in a jagged array, given row and object masks
  - `ha.max_in_offsets(offsets, content, mask_rows, mask_content)`: as above, but find the maximum
  - `ha.prod_in_offsets(offsets, content, mask_rows, mask_content, dtype=None)`: compute the product in a jagged array
  - `ha.set_in_offsets(content, offsets, indices, target, mask_rows, mask_content)`: set the indexed value in a jagged array to a target
  - `ha.get_in_offsets(offsets, content, indices, mask_rows, mask_content)`:   retrieve the indexed values in a jagged array, e.g. get the leading jet pT
  - `ha.compute_new_offsets(offsets_old, mask_objects, offsets_new)`: given an   awkward offset array and a mask, create an offset array of the unmasked elements
  - `ha.searchsorted(bins, vals, side="left")`: 1-dimensional search in a sorted array
  - `ha.histogram_from_vector(data, weights, bins, mask=None)`: fill a 1-dimensional weighted histogram with arbitrary sorted bins, possibly using a mask
  - `ha.histogram_from_vector_several(variables, weights, mask)`: fill several   histograms simultaneously based on `variables=[(data0, bins0), ...]`
  - `ha.get_bin_contents(values, edges, contents, out)`: look up the bin contents of   a histogram based on a vector of values 
  - `ha.select_opposite_sign(muons, in_mask)`: select the first pair with opposite sign charge
  - `ha.mask_deltar_first(objs1, mask1, objs2, mask2, drcut)`: given two collections of objects defined by eta, phi and offsets, mask the objects in the first collection that satisfy `DeltaR(o1, o2) < drcut)`

## Usage

This is a minimal example from [examples/simple_hzz.py](../blob/master/examples/simple_hzz.py), which can be run from this repository directly using
~~~
PYTHONPATH=. python3 examples/simple_hzz.py
PYTHONPATH=. HEPACCELERATE_CUDA=1 python3 examples/simple_hzz.py
~~~

```python
#usr/bin/env python3
#Run as PYTHONPATH=. python3 examples/simple_hzz.py

#In case you use CUDA, you may have to find the libnvvm.so on your system manually
import os
os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda/nvvm/lib64/libnvvm.so"
os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda/nvvm/libdevice/"
import numba

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

#choose whether or not to use the GPU backend
use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
NUMPY_LIB, ha = choose_backend(use_cuda=use_cuda)

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

dataset = Dataset("HZZ", [filename], datastructures, treename="events")

#load data to memory
dataset.load_root()

#process data
results = dataset.analyze(analyze_data_function, verbose=True, parameters={"muons_ptcut": 30.0})
results.save_json("out.json")
```

A more complete CMS analysis example an be found in [analysis_hmumu.py](https://github.com/hepaccelerate/hepaccelerate-cms/blob/master/tests/hmm/analysis_hmumu.py). Currently, for simplicity and in the spirit of prototyping, that repository comes batteries-included with CMS-specific analysis methods.

## Recommendations on data locality and remote data
In order to make full use of modern CPUs or GPUs, you want to bring the data as close as possible to where the work is done, otherwise you will spend most of the time waiting for the data to arrive rather than actually performing the computations.

With CMS NanoAOD with event sizes of 1-2 kB/event, 1 million events is approximately 1-2 GB on disk. Therefore, you can fit a significant amount of data used in a HEP analysis on a commodity SSD. In order to copy the data to your local disk, use grid tools such as `gfal-copy` or even `rsync` to fetch it from your nearest Tier2. Preserving the filename structure (`/store/...`) will allow you to easily run the same code on multiple sites.

## Frequently asked questions

 - *Why are you doing this array-based analysis business?* Mainly out of curiosity, and I could not find a tool available with which I could do HEP analysis on data on a local disk with MHz rates. It is possible that dask/spark/RDataFrame will soon work well enough for this purpose, but until then, I can justify writing a few functions.
 - *How does this relate to the awkward-array project?* We use the jagged structure provided by the awkward arrays, but implement common HEP functions such as deltaR matching as parallelizable loops or 'kernels' running directly over the array contents, taking into account the event structure. We make these loops fast with Numba, but allow you to debug them by going back to standard python when disabling the compilation.
 - *How does this relate to the coffea/fnal-columnar-analysis-tools project?* It's very similar, you should check out that project! We implement less methods, mostly by explicit loops in Numba, and on GPUs as well as CPUs.
 - *Why don't you use the array operations (`JaggedArray.sum`, `argcross` etc) implemented in awkward-array?* They are great! However, in order to easily use the same code on either the CPU or GPU, we chose to implement the most common operations explicitly, rather than relying on numpy/cupy to do it internally. This also seems to be faster, at the moment.
 - *What if I don't have access to a GPU?* You should still be able to see event processing speeds in the hundreds of kHz to a few MHz for common analysis tasks.
 - *How do I plot my histograms that are saved in the output JSON?* Load the JSON contents and use the `edges` (left bin edges, plus last rightmost edge), `contents` (weighted bin contents) and `contents_w2` (bin contents with squared weights, useful for error calculation) to access the data directly.
 - *I'm a GPU programming expert, and I worry your CUDA kernels are not optimized. Can you comment?* Good question! At the moment, they are indeed not very optimized, as we do a lot of control flow (`if` statements) in them. However, the GPU analysis is still about 2x faster than a pure CPU analysis, as the CPU is more free to work on loading the data, and this gap is expected to increase as the analysis becomes more complicated (more systematics, more templates). At the moment, we see pure GPU processing speeds of about 8-10 MHz for in-memory data, and data loading from cache at about 4-6 MHz. Have a look at the nvidia profiler results [nvprof1](profiling/nvprof1.png), [nvprof2](profiling/nvprof2.png) to see what's going on under the hood. Please give us a hand to make it even better!
 - *What about running this code on multiple machines?* You can do that, currently just using usual batch tools, but we are looking at other ways (dask, joblib, spark) to distribute the analysis across multiple machines. 
 - *What about running this code on data that is remote (XROOTD)?* You can do that thanks to the `uproot` library, but then you gain very little benefit from having a fast CPU or GPU, as you will spend most of your time just waiting for input.
 - *How much RAM is needed?* The amount of RAM determines how much data can be preloaded to memory. You can either process data in memory all at once, which makes rerunning very fast, or set up a batched pipeline. In case of the batched pipeline, no more than a few GB/thread of RAM is needed, and overall processing speeds are still around the MHz-level.
