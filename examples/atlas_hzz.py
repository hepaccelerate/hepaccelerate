# usr/bin/env python3
# Run as PYTHONPATH=. python3 examples/simple_hzz.py

# In case you use CUDA, you may have to find the libnvvm.so on your system manually
import os, time, glob, argparse, multiprocessing
import numba
import sys
import numpy as np
import uproot
import hepaccelerate
import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import hepaccelerate.backend_cpu as backend_cpu
import matplotlib
import infofile
import json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import wasserstein_distance
from plot_utils import plot_hist_ratio

ha = None
lumi =10.0

# define our analysis function
def analyze_data_function(data, parameters):
    ret = Results()
    ha = parameters["ha"]
    num_events = data["num_events"]
    lep = data["Lep"]
    lep.hepaccelerate_backend = ha
    lep.attrs_data["pt"] = lep.lep_pt    
    lep.attrs_data["eta"] = lep.lep_eta
    lep.attrs_data["phi"] = lep.lep_phi
    lep.attrs_data["charge"] = lep.lep_charge
    lep.attrs_data["type"] = lep.lep_type
    
    lep_mass = np.zeros_like(lep["pt"],dtype=nplib.float32)
    lep_mass = np.where(lep["type"]==11,0.511,lep_mass)
    lep_mass = np.where(lep["type"]==13,105.65837,lep_mass)

    lep.attrs_data["mass"] = lep_mass
    mask_events = nplib.ones(lep.numevents(), dtype=nplib.bool)
    
    num_lep_event = kernels.sum_in_offsets(
        backend,
        lep.offsets,
        lep.masks["all"],
        mask_events,
        lep.masks["all"],
        nplib.int8,
    )
    mask_events_4lep = num_lep_event == 4
    lep_attrs = [ "pt", "eta", "phi", "charge","type","mass"]#, "ptcone30", "etcone20"]
    
    #ximport pdb; pdb.set_trace();
    lep0 = lep.select_nth(0, mask_events_4lep, lep.masks["all"], attributes=lep_attrs)
    lep1 = lep.select_nth(1, mask_events_4lep, lep.masks["all"], attributes=lep_attrs)
    lep2 = lep.select_nth(2, mask_events_4lep, lep.masks["all"], attributes=lep_attrs)
    lep3 = lep.select_nth(3, mask_events_4lep, lep.masks["all"], attributes=lep_attrs)
    
    mask_event_sumchg_zero = (lep0["charge"]+lep1["charge"]+lep2["charge"]+lep3["charge"] == 0) 
    sum_lep_type = lep0["type"]+lep1["type"]+lep2["type"]+lep3["type"] 
    
    mask_event_sum_lep_type = np.logical_or((sum_lep_type == 44),np.logical_or((sum_lep_type == 48),(sum_lep_type == 52) ) )
    mask_events = mask_events & mask_event_sumchg_zero & mask_events_4lep & mask_event_sum_lep_type
    

    mask_lep1_passing_pt = lep1["pt"] > parameters["leading_lep_ptcut"]
    mask_lep2_passing_pt = lep2["pt"] > parameters["lep_ptcut"]
    
    mask_events = mask_events & mask_lep1_passing_pt & mask_lep2_passing_pt

    l0 = to_cartesian(lep0)
    l1 = to_cartesian(lep1)
    l2 = to_cartesian(lep2)
    l3 = to_cartesian(lep3)

    llll = {k: l0[k] + l1[k] + l2[k] + l3[k] for k in ["px", "py", "pz", "e"]}

    llll_sph = to_spherical(llll)

    llll_sph["mass"] = llll_sph["mass"]/1000. # Convert to GeV
    
    #import pdb;pdb.set_trace();
    # compute a weighted histogram
    weights = nplib.ones(num_events, dtype=nplib.float32)
    ## Add xsec weights based on sample name
    if parameters["is_mc"]:
        weights = data['eventvars']['mcWeight']*data['eventvars']['scaleFactor_PILEUP']*data['eventvars']['scaleFactor_ELE']*data['eventvars']['scaleFactor_MUON']*data['eventvars']['scaleFactor_LepTRIGGER']
        info = infofile.infos[parameters["sample"]]
        weights *= (lumi*1000*info["xsec"])/(info["sumw"]*info["red_eff"])
    
    bins = nplib.linspace(110, 150, 21, dtype=nplib.float32)
    hist_m4lep= Histogram(
        *kernels.histogram_from_vector(
            backend,
            llll_sph["mass"][mask_events],
            weights[mask_events],
            bins,
        )
    )
    # save it to the output
    ret["hist_m4lep"] = hist_m4lep
    return ret

def to_cartesian(arrs):
    pt = arrs["pt"]
    eta = arrs["eta"]
    phi = arrs["phi"]
    mass = arrs["mass"]
    px, py, pz, e = backend.spherical_to_cartesian(pt, eta, phi, mass)
    return {"px": px, "py": py, "pz": pz, "e": e}

def rapidity(e, pz):
    return 0.5*np.log((e + pz) / (e - pz))

"""
Given a a dictionary of arrays of cartesian coordinates (px, py, pz, e),
computes the array of spherical coordinates (pt, eta, phi, m)
    arrs: dict of str -> array
    returns: dict of str -> array
"""
def to_spherical(arrs):
    px = arrs["px"]
    py = arrs["py"]
    pz = arrs["pz"]
    e = arrs["e"]
    pt, eta, phi, mass = backend.cartesian_to_spherical(px, py, pz, e)
    rap = rapidity(e, pz)
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass, "rapidity": rap}

def pct_barh(ax, values, colors):
    prev = 0
    norm = sum(values)
    for v, c in zip(values, colors):
        ax.barh(0, width=v/norm, height=1.0, left=prev, color=c)
        prev += v/norm
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0,prev)
    ax.axis('off')

def make_plot(datasets):
    res = {}
    mc_samples =[]
    for ds, fn_pattern, is_mc in datasets:
        with open("data/atlas/{0}.json".format(ds)) as f:
            ret = json.load(f)["hist_m4lep"]
            
            res[ds] = hepaccelerate.Histogram(ret["contents"], ret["contents_w2"], ret["edges"])
            
            #remove the overflow bin
            res[ds].contents[-1] = 0
            res[ds].contents_w2[-1] = 0
            
            if 'data' in ds:
                hd = res[ds]
            else:
                mc_samples += [ds]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if np.sum(hd.contents) == 0:
        print("ERROR: Histogram was empty, skipping")
        return

    htot_nominal = copy.deepcopy(hd)
    htot_nominal.contents[:] = 0
    htot_nominal.contents_w2[:] = 0

    hmc = []
    for mc_samp in mc_samples:
        h = res[mc_samp]
        h.label = nice_names.get(mc_samp, mc_samp)
        h.color = colors[mc_samp][0]/255.0, colors[mc_samp][1]/255.0, colors[mc_samp][2]/255.0
        hmc += [h]
            
    #hmc = [hmc_g[k[0]] for k in groups]
    for h in hmc:
        htot_nominal.contents += h.contents
        htot_nominal.contents_w2 += h.contents_w2
    hd.label = "data ({0:.1E})".format(np.sum(hd.contents))

    extra_kwargs = {
    "hist_m4lep": {
        "do_log": True,
        "ylim": (0, 100),
        "xlim": (110.,150.)
    }}

    figure = plt.figure(figsize=(5,5))
    a1, a2 = plot_hist_ratio(
        hmc, hd,
        figure=figure, **extra_kwargs)
    
#     colorlist = [h.color for h in hmc]
#     a1inset = inset_axes(a1, width=1.0, height=0.1, loc=2)

#     pct_barh(a1inset, [np.sum(h.contents) for h in hmc], colorlist)
#     #a2.grid(which="both", linewidth=0.5)

    # Ratio axis ticks
    #ts = a2.set_yticks([0.5, 1.0, 1.5], minor=False)
    #ts = a2.set_yticks(np.arange(0,2,0.2), minor=True)
    #ts = a2.set_xticklabels([])

    a1.text(0.03,0.95, "Atlas Open Data \n" +
        r"$L = {0:.1f}\ fb^{{-1}}$".format(lumi),
        #"\nd/mc={0:.2f}".format(np.sum(hd["contents"])/np.sum(htot_nominal["contents"])) +
        #"\nwd={0:.2E}".format(wasserstein_distance(htot_nominal["contents"]/np.sum(htot_nominal["contents"]), hd["contents"]/np.sum(hd["contents"]))),
        horizontalalignment='left',
        verticalalignment='top',
        transform=a1.transAxes,
        fontsize=10
    )
    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles[::-1], labels[::-1], frameon=False, fontsize=10, loc=1, ncol=2)

    #a1.set_title(catname + " ({0})".format(analysis_names[analysis][datataking_year]))
    a2.set_xlabel(r'$M_{4l}$ (GeV)')

    binwidth = np.diff(hd.edges)[0]
    a1.set_ylabel("Events / [{0:.1f} GeV]".format(binwidth))

    if not os.path.isdir("paper/plots/atlas"):
        os.makedirs("paper/plots/atlas") 
    plt.savefig("paper/plots/atlas/m_4lep.pdf", bbox_inches="tight")
    plt.savefig("paper/plots/atlas/m_4lep.png", bbox_inches="tight", dpi=100)
    plt.close(figure)
    del figure

    return

datasets = [
    ("Zee", "Atlas_opendata/mc_361106.Zee.4lep.root", True),
    ("Zmumu", "Atlas_opendata/mc_*Zmumu.4lep.root", True), 
    ("ttbar_lep", "Atlas_opendata/mc_*ttbar*.4lep.root", True),
    ("llll", "Atlas_opendata/mc_*llll*.4lep.root", True),
    ('ggH125_ZZ4lep',"Atlas_opendata/mc_*ggH125_ZZ4lep.4lep.root", True),
    ('VBFH125_ZZ4lep',"Atlas_opendata/mc_*VBFH125_ZZ4lep.4lep.root", True),
    ('WH125_ZZ4lep',"Atlas_opendata/mc_*WH125_ZZ4lep.4lep.root", True),
    ('ZH125_ZZ4lep',"Atlas_opendata/mc_*ZH125_ZZ4lep.4lep.root", True),
    ("data","Atlas_opendata/data*.4lep.root",False)
]

colors = {
    "Zee": (254, 254, 83),
    "Zmumu": (109, 253, 245),
    "ttbar_lep": (67, 150, 42),
    "llll": (247, 206, 205),
    "ggH125_ZZ4lep": (0, 0, 0),
    "VBFH125_ZZ4lep": (0, 0, 0),
    "WH125_ZZ4lep": (0, 0, 0),
    "ZH125_ZZ4lep": (0, 0, 0),
}

nice_names = {
    "Zee": r"$Z \rightarrow ee$"
}

if __name__ == "__main__":


    use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
    # choose whether or not to use the GPU backend
    if use_cuda:
        import setGPU

    nplib, backend = choose_backend(use_cuda=use_cuda)
        
    # Predefine which branches to read from the TTree and how they are grouped to objects
    # This will be verified against the actual ROOT TTree when it is loaded
    datastructures = {
        "Lep": [
            ("lep_pt", "float32"),
            ("lep_eta","float32"),
            ("lep_phi","float32"),
            ("lep_charge","int32"),
            ("lep_type","uint32"),
            ("lep_ptcone30","float32"),
            ("lep_etcone20","float32")
            
        ],
        "EventVariables": [
            ("lep_n", "int32"),
            ("mcWeight", "float32"),
            ("scaleFactor_PILEUP", "float32"),
            ("scaleFactor_ELE", "float32"),
            ("scaleFactor_MUON", "float32"),
            ("scaleFactor_LepTRIGGER", "float32")
        ],
    }
    

    # Load this input file
    #filename = "data/data_A.4lep.root"
    if not os.path.isdir("data/atlas"):
        os.makedirs("data/atlas")

    walltime_t0 = time.time()
    for ds, fn_pattern, is_mc in datasets:
        filename = glob.glob(fn_pattern)
        print(filename)
        if len(filename) == 0:
            raise Exception(
                "Could not find any filenames for dataset={0}: {{fn_pattern}}={1}".format(
                    ds, fn_pattern
                )
            )

        # Define a dataset, given the data structure and a list of filenames
        dataset = Dataset(ds, filename, datastructures, treename="mini")
        # Load the ROOT files
        dataset.load_root(verbose=True)
    
        # merge arrays across files into one big array
        dataset.merge_inplace(verbose=True)
    
        # move to GPU if CUDA was specified
        dataset.move_to_device(nplib, verbose=True)
            
        # process data, save output as a json file
        results = dataset.analyze(
            analyze_data_function, verbose=True, parameters={
                "lep_ptcut": 10000.0, #MeV units
                "leading_lep_ptcut": 15000.0, #MeV units
                "sample": ds,
                "is_mc": is_mc,
                "ha":backend
            }
        )
        results.save_json("data/atlas/{0}.json".format(ds))

    make_plot(datasets)
