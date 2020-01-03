import numpy
import uproot

import hepaccelerate.kernels as kernels
from hepaccelerate import backend_cpu

tt = uproot.open("data/HZZ.root").get("events")

mu_px = tt.array("Muon_Px")
offsets = mu_px.offsets
pxs = mu_px.content

sel_ev = numpy.ones(len(tt), dtype=numpy.bool)
sel_mu = numpy.ones(len(pxs), dtype=numpy.bool)

#This is the same functionality as awkward.array.max, but supports either CPU or GPU!
#Note that events with no entries will be filled with zeros rather than skipped
event_max_px = kernels.max_in_offsets(
    backend_cpu,
    offsets,
    pxs,
    sel_ev,
    sel_mu)

event_max_px_awkward = mu_px.max()
event_max_px_awkward[numpy.isinf(event_max_px_awkward)] = 0

print(numpy.all(event_max_px_awkward == event_max_px))
