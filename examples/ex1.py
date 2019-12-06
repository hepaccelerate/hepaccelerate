import numpy
import uproot

from hepaccelerate import backend_cpu as ha

tt = uproot.open("data/HZZ.root").get("events")

mu_px = tt.array("Muon_Px")
offsets = mu_px.offsets
pxs = mu_px.content

sel_ev = numpy.ones(len(tt), dtype=numpy.bool)
sel_mu = numpy.ones(len(pxs), dtype=numpy.bool)

event_max_pt = ha.max_in_offsets(
    offsets,
    pxs,
    sel_ev,
    sel_mu)
