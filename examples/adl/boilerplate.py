#In case you use CUDA, you may have to find the libnvvm.so on your system manually
import os, sys
import numba
import numpy as np

import hepaccelerate
import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

def run(analyze_data_function, datastructures, outfile, additional_parameters={}):
    #choose whether or not to use the GPU backend
    use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
    if use_cuda:
        import setGPU
    
    nplib, backend = choose_backend(use_cuda=use_cuda)
    backend.nplib = nplib
    
    #Define a dataset, given the data structure and a list of filenames 
    dataset = Dataset("SingleMu", ["Run2012B_SingleMu.root"], datastructures, treename="Events")
   
    #Load the ROOT files 
    dataset.load_root(verbose=True)
    
    #merge arrays across files into one big array
    dataset.merge_inplace(verbose=True)
    
    #move to GPU if CUDA was specified
    dataset.move_to_device(nplib, verbose=True)
    
    #process data, save output as a json file
    results = dataset.analyze(analyze_data_function, verbose=True,
        parameters={"backend": backend, "nplib": nplib, **additional_parameters}
    )
    results.save_json(outfile)
