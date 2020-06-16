#usr/bin/env python3
#Run as PYTHONPATH=. python3 examples/adl/ex03.py

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

    jets = data["Jet"]
    mask_all_jets = nplib.ones(jets.numobjects(), dtype=nplib.bool)

    jets_passing_pt = data["Jet"]["pt"] > parameters["ptcut"]

    num_jets_in_event = kernels.sum_in_offsets(backend,
        jets.offsets, #jet jaggedness offsets
        jets_passing_pt, #sum the values of this array within the offsets
        mask_events, #consider only events passing this mask
        mask_all_jets, #consider only jets passing this mask, by default, all jets
        nplib.float32 #output datatype
    )

    msk_ev = num_jets_in_event >= 2

    weights = nplib.ones(num_events, dtype=nplib.float32)
    bins = nplib.linspace(0, 300, 101, dtype=nplib.float32)
    ret["hist_met"] = Histogram(*kernels.histogram_from_vector(
        backend, data["EventVariables"]["MET_sumet"][msk_ev], weights[msk_ev], bins))

    return ret

datastructures = {
   "Jet": [
       ("Jet_pt", "float32"),
       ("Jet_eta", "float32"),
    ],
   "EventVariables": [
       ("MET_sumet", "float32"),
   ]
}
   
if __name__ == "__main__":
    run(analyze_data_function, datastructures, "out_ex01.json", {"ptcut": 40})
