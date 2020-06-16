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

    num_events = data["num_events"]
    weights = nplib.ones(num_events, dtype=nplib.float32)
    bins = nplib.linspace(0,300,101, dtype=nplib.float32)

    ret["hist_met"] = Histogram(*kernels.histogram_from_vector(
        backend, data["EventVariables"]["MET_sumet"], weights, bins))

    return ret

datastructures = {
   "EventVariables": [
       ("MET_sumet", "float32"),
   ]
}
   
if __name__ == "__main__":
    run(analyze_data_function, datastructures, "out_ex01.json")
