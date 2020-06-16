#usr/bin/env python3
#Run as PYTHONPATH=. python3 examples/adl/ex01.py

import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Histogram
from boilerplate import run

#define our analysis function
def analyze_data_function(data, parameters):
    backend = parameters["backend"]
    nplib = parameters["nplib"]

    ret = Results()

    num_jets = data["Jet"].numobjects()
    weights = nplib.ones(num_jets, dtype=nplib.float32)
    bins = nplib.linspace(0, 300, 101, dtype=nplib.float32)
    
    ret["hist_jet_pt"] = Histogram(*kernels.histogram_from_vector(
        backend, data["Jet"]["pt"], weights, bins))

    return ret

datastructures = {
   "Jet": [
       ("Jet_pt", "float32"),
    ],
   "EventVariables": [],
}
   
if __name__ == "__main__":
    run(analyze_data_function, datastructures, "out_ex01.json")
