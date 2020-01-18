[![GitHub Actions Status: CI](https://github.com/hepaccelerate/hepaccelerate/workflows/CI/CD/badge.svg)](https://github.com/hepaccelerate/hepaccelerate/actions?query=workflow%3ACI%2FCD+branch%3Amaster)
[![Build Status](https://travis-ci.com/hepaccelerate/hepaccelerate.svg?branch=master)](https://travis-ci.com/hepaccelerate/hepaccelerate)
[![pipeline status](https://gitlab.cern.ch/jpata/hepaccelerate/badges/master/pipeline.svg)](https://gitlab.cern.ch/jpata/hepaccelerate/commits/master)
[![DOI](https://zenodo.org/badge/191644111.svg)](https://zenodo.org/badge/latestdoi/191644111)

# hepaccelerate

- Fast kernels for HEP data analysis with [jagged arrays](https://github.com/scikit-hep/awkward-array) using python + [Numba](http://numba.pydata.org/)
- Use **any ntuple**, as long as you can open it with [uproot](https://github.com/scikit-hep/uproot)
- analyze a billion events with systematic to histograms in minutes on a single workstation
  - 1e9 events / (50 kHz x 24 CPU threads) ~ 13 minutes
- weighted histograms, deltaR matching and [more](https://github.com/hepaccelerate/hepaccelerate#kernels)
- use a CPU or an nVidia CUDA GPU with the same interface!
- this is **not** an analysis framework, but rather a set of example helper functions for fast jagged array processing

**Under active development and use by a few CMS analyses!**

<p float="left">
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/paper/plots/kernel_speedup.png" alt="Kernel benchmarks" width="300"/>
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/paper/plots/analysis_benchmark.png" alt="Analysis benchmarks" width="300"/>
</p>

More details are available:
- writeup: https://arxiv.org/abs/1906.06242v2
- PyHEP 2019 talk: https://indico.cern.ch/event/833895/contributions/3577804/attachments/1927026/3192574/2019_10_15_pyhep.pdf

## Installation

The library can be installed using `pip` in python >3.6:
```bash
pip install hepaccelerate
```

You may also clone this library as a part of your project, in which case you will need the following libraries:
 - `pip install uproot numba xxhash lz4`

Optional libraries, which may be easier to install with conda:
 - `cupy` for GPU support
 - `cudatoolkit` for GPU support
 - `dask` for running the large-scale data analysis example
 - `xxhash` for LZ4 support

## Documentation
This code consists of two parts which can be used independently:
  - the accelerated HEP kernels that run on jagged data: [kernels.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/kernels.py)
    - CPU backend, [backend_cpu.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cpu.py)
    - CUDA backend, [backend_cuda.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cuda.py)
  - JaggedStruct, Dataset and Histogram classes to help with HEP dataset management, [utils.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/utils.py)

### Environment variables

The following environment variables can be used to tune the number of threads:
```
HEPACCELERATE_CUDA=0 #1 to enable CUDA
NUMBA_NUM_THREADS=1 #number of parallel threads for numba CPU kernels
```

### Kernels

The jagged kernels work on the basis of the `content` and `offsets` arrays based on `awkward.JaggedArray` and can be used on `numpy` or `cupy` data arrays. The full list of kernels is available in [kernels.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/kernels.py).

We have implemented the following kernels for both the CPU and CUDA backends:
  - `searchsorted`: 1-dimensional search in a sorted array as `np.searchsorted`
  - `histogram_from_vector`: fill a 1-dimensional weighted histogram
  - `histogram_from_vector_several`: fill several 1-dimensional histograms simultaneously
  - `get_bin_contents`: look up the bin contents of a histogram based on a vector of values
  - `compute_new_offsets`: given an offset array and a mask, create an offset array of the unmasked elements
  - `select_opposite_sign`: select the first pair with opposite sign charge
  - `mask_deltar_first`: given two collections of objects defined by eta, phi and offsets, mask the objects in the first collection that satisfy `DeltaR(o1, o2) < drcut)`
  - `min_in_offsets`: retrieve the minimum value in a jagged array
  - `max_in_offsets`: as above, but find the maximum
  - `prod_in_offsets`: compute the product in a jagged array
  - `sum_in_offsets`: compute the sum in a jagged array
  - `set_in_offsets`: set the indexed value in a jagged array to a target value
  - `get_in_offsets`:   retrieve the indexed values in a jagged array

The kernels can be used as follows:
```python
import numpy
import uproot

from hepaccelerate import backend_cpu as ha

tt = uproot.open("data/HZZ.root").get("events")

mu_px = tt.array("Muon_Px")
offsets = mu_px.offsets
pxs = mu_px.content

sel_ev = numpy.ones(len(tt), dtype=numpy.bool)
sel_mu = numpy.ones(len(pxs), dtype=numpy.bool)

#This is the same functionality as awkward.array.max, but supports either CPU or GPU!
#Note that events with no entries will be filled with zeros rather than skipped
event_max_px = kernels.max_in_offsets(
    backend, 
    offsets,
    pxs,
    sel_ev,
    sel_mu)

event_max_px_awkward = mu_px.max()
event_max_px_awkward[numpy.isinf(event_max_px_awkward)] = 0

print(numpy.all(event_max_px_awkward == event_max_px))
```

### Dataset utilities

  - `Dataset(name, filenames, datastructures, datapath="", treename="Events", is_mc=True)`: represents a dataset of many jagged arrays from multiple files with the same structure in memory
    - `load_root()`: Load the dataset from ROOT files to memory
    - `structs[name][ifile]`: JaggedStruct `name` in file `ifile`
    - `compact(masks)`: Drop events that do not pass the masks, one per file
  - `JaggedStruct`: Container for multiple singly-nested jagged arrays with the same offsets
    - `getattr(name)`: get the content array corresponding to an attribute (e.g. `jet.pt`)
    - `offsets`: get the offsets array
    - `move_to_device(array_lib)`: with `array_lib` being either `numpy` or `cupy`
  - `Histogram(contents, contents_w2, edges)`: a very simple container for a one-dimensional histogram

The following example illustrates how the dataset structures are used:
```python
from hepaccelerate.utils import Dataset

#Define which columns we want to access
datastructures = {
    "Muon": [
        ("Muon_Px", "float32"),
        ("Muon_Py", "float32"),
    ],
    "Jet": [
        ("Jet_E", "float32"),
        ("Jet_btag", "float32"),
    ],
    "EventVariables": [
        ("NPrimaryVertices", "int32"),
        ("triggerIsoMu24", "bool"),
        ("EventWeight", "float32")
    ]
}

#Define the dataset across the files
dataset = Dataset("HZZ", ["data/HZZ.root"], datastructures, treename="events", datapath="")

#Load the data to memory
dataset.load_root()

#Jets in the first file
ifile = 0
jets = dataset.structs["Jet"][ifile]

#common offset array for jets
jets_offsets = jets.offsets
print(jets_offsets)

#data arrays
jets_energy = jets.E
jets_btag = jets.btag
print(jets_energy)
print(jets_btag)

ev_weight = dataset.eventvars[ifile]["EventWeight"]
print(ev_weight)
```

The kernels can be used as follows:
```python
import numpy
import uproot

from hepaccelerate import backend_cpu as ha

tt = uproot.open("data/HZZ.root").get("events")

mu_px = tt.array("Muon_Px")
offsets = mu_px.offsets
pxs = mu_px.content

sel_ev = numpy.ones(len(tt), dtype=numpy.bool)
sel_mu = numpy.ones(len(pxs), dtype=numpy.bool)

#This is the same functionality as awkward.array.max, but supports either CPU or GPU!
#Note that events with no entries will be filled with zeros rather than skipped
event_max_px = ha.max_in_offsets(
    offsets,
    pxs,
    sel_ev,
    sel_mu)

event_max_px_awkward = mu_px.max()
event_max_px_awkward[numpy.isinf(event_max_px_awkward)] = 0

print(numpy.all(event_max_px_awkward == event_max_px))
```

### Dataset utilities

  - `Dataset(name, filenames, datastructures, datapath="", treename="Events", is_mc=True)`: represents a dataset of many jagged arrays from multiple files with the same structure in memory
    - `load_root()`: Load the dataset from ROOT files to memory
    - `structs[name][ifile]`: JaggedStruct `name` in file `ifile`
    - `compact(masks)`: Drop events that do not pass the masks, one per file
  - `JaggedStruct`: Container for multiple singly-nested jagged arrays with the same offsets
    - `getattr(name)`: get the content array corresponding to an attribute (e.g. `jet.pt`)
    - `offsets`: get the offsets array
    - `move_to_device(array_lib)`: with `array_lib` being either `numpy` or `cupy`
  - `Histogram(contents, contents_w2, edges)`: a very simple container for a one-dimensional histogram

The following example illustrates how the dataset structures are used:
```python
from hepaccelerate.utils import Dataset

#Define which columns we want to access
datastructures = {
    "Muon": [
        ("Muon_Px", "float32"),
        ("Muon_Py", "float32"),
    ],
    "Jet": [
        ("Jet_E", "float32"),
        ("Jet_btag", "float32"),
    ],
    "EventVariables": [
        ("NPrimaryVertices", "int32"),
        ("triggerIsoMu24", "bool"),
        ("EventWeight", "float32")
    ]
}

#Define the dataset across the files
dataset = Dataset("HZZ", ["data/HZZ.root"], datastructures, treename="events", datapath="")

#Load the data to memory
dataset.load_root()

#Jets in the first file
ifile = 0
jets = dataset.structs["Jet"][ifile]

#common offset array for jets
jets_offsets = jets.offsets
print(jets_offsets)

#data arrays
jets_energy = jets.E
jets_btag = jets.btag
print(jets_energy)
print(jets_btag)

ev_weight = dataset.eventvars[ifile]["EventWeight"]
print(ev_weight)
```

## Usage

A minimal example can be found in [examples/simple_hzz.py](../blob/master/examples/simple_hzz.py), which can be run from this repository directly using

```bash
#required just for the example
pip install matplotlib

#on CPU

PYTHONPATH=. python3 examples/simple_hzz.py

#on GPU
PYTHONPATH=. HEPACCELERATE_CUDA=1 python3 examples/simple_hzz.py
```

## Full example analysis

For an example top quark pair analysis on ~144GB of CMS Open Data, please see [full_analysis.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/examples/full_analysis.py). This analysis uses [dask](https://dask.org/) to run many parallel processes either on the CPU or GPU. We stress that this is purely an example and using dask is by no means required to use the kernels.

```
#make sure the libraries required for the example are installed
pip install tensorflow==1.15 keras dask distributed

#Download the large input dataset, need ~150GB of free space in ./
./examples/download_example_data.sh ./

#Starts a dask cluster and runs the analysis on all CPUs
PYTHONPATH=. HEPACCELERATE_CUDA=0 python3 examples/full_analysis.py --out data/out.pkl --datapath ./ --dask-server ""
```

The following methods are implemented using both the CPU and GPU backends:
- event and object selection
- on-the-fly variation of jets using mockup jet energy corrections
- reconstruction of the jet triplet with invariant mass closest to the top quark mass
- signal-to-background DNN evaluation
- filling around 20 control histograms with systematic variations

<p float="left">
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/paper/plots/sumpt.png" alt="Top quark pair analysis" width="300"/>
</p>

## Recommendations on data locality and remote data
In order to make full use of modern CPUs or GPUs, you want to bring the data as close as possible to where the work is done, otherwise you will spend most of the time waiting for the data to arrive rather than actually performing the computations.

With CMS NanoAOD with event sizes of 1-2 kB/event, 1 million events is approximately 1-2 GB on disk. Therefore, you can fit a significant amount of data used in a HEP analysis on a commodity SSD.
In order to copy the data to your local disk, use grid tools such as `gfal-copy` or even `rsync` to fetch it from your nearest Tier2. Preserving the filename structure (`/store/...`) will allow you to easily run the same code on multiple sites.

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
