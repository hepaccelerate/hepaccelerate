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
  <img src="https://github.com/hepaccelerate/hepaccelerate/blob/master/paper/plots/kernel_benchmarks.png" alt="Kernel benchmarks" width="300"/>
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

You may also clone this library as a part of your project, in which case you will need:
 - `pip install uproot numba xxhash lz4`

Optional libraries, which may be easier to install with conda:
 - `cupy` for GPU support
 - `cudatoolkit` for GPU support
 - `dask` for running the large-scale example
 - `xxhash` for LZ4 support

## Documentation
This code consists of two parts which can be used independently:
  - the accelerated HEP kernels that run on jagged data:
    - multithreaded CPU: [backend_cpu.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cpu.py)
    - CUDA GPU: [backend_cuda.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/hepaccelerate/backend_cuda.py)  
  - JaggedStruct, Dataset and Histogram classes to help with HEP dataset management

## Kernels

The jagged kernels work on the basis of the `content` and `offsets` arrays based on `awkward.JaggedArray`.

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

```bash
PYTHONPATH=. python3 examples/simple_hzz.py

#on GPU
PYTHONPATH=. HEPACCELERATE_CUDA=1 python3 examples/simple_hzz.py
```

## Example analysis
For an example top quark pair analysis on ~200GB of CMS Open Data, please see [full_analysis.py](https://github.com/hepaccelerate/hepaccelerate/blob/master/examples/full_analysis.py). The following methods are implementd using both the CPU and GPU backends:
- event and object selection
- on-the-fly variation of jets using mockup jet energy corrections
- reconstruction of the jet triplet with invariant mass closest to the top quark mass
- signal-to-background DNN evaluation
- filling around 20 control histograms with systematic variations

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
