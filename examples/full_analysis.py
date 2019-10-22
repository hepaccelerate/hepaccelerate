import os, glob, sys, time, argparse, multiprocessing
import pickle, math, requests
from collections import OrderedDict

import uproot
import hepaccelerate
from hepaccelerate.utils import Histogram, Results

import numba
from numba import cuda
import numpy as np

use_cuda = bool(int(os.environ.get("HEPACCELERATE_CUDA", 0)))

#save training arrays for DNN
save_arrays = False

#GPUs to use when multiprocessing
gpu_id_list = [0, 1]

#just to load the DNN model files
def download_file(filename, url):
    """
    Download an URL to a file
    """
    print("downloading {0}".format(url))
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        # Write response data to file
        iblock = 0
        for block in response.iter_content(4096):
            if iblock % 1000 == 0:
                sys.stdout.write(".");sys.stdout.flush()
            iblock += 1
            fout.write(block)
    print("download complete!")

def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        download_file(filename, url)
        return True
    return False

#DNN weights produced using examples/train_dnn.py and setting save_arrays=True
class DNNModel:
    def __init__(self):
        import keras
        self.models = []
        for i in range(1):
            self.models += [keras.models.load_model("data/model_kf{0}.h5".format(i))]

    def eval(self, X, use_cuda):
        if use_cuda:
            X = NUMPY_LIB.asnumpy(X)
        rets = [NUMPY_LIB.array(m.predict(X, batch_size=10000)[:, 0]) for m in self.models]
        return rets

def create_datastructure(ismc):
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"),
            ("Muon_eta", "float32"),
            ("Muon_phi", "float32"),
            ("Muon_mass", "float32"),
            ("Muon_charge", "int32"),
            ("Muon_pfRelIso03_all", "float32"),
            ("Muon_tightId", "bool")
        ],
        "Electron": [
            ("Electron_pt", "float32"),
            ("Electron_eta", "float32"),
            ("Electron_phi", "float32"),
            ("Electron_mass", "float32"),
            ("Electron_charge", "int32"),
            ("Electron_pfRelIso03_all", "float32"),
            ("Electron_pfId", "bool")
        ],
        "Jet": [
            ("Jet_pt", "float32"),
            ("Jet_eta", "float32"),
            ("Jet_phi", "float32"),
            ("Jet_mass", "float32"),
            ("Jet_btag", "float32"),
            ("Jet_puId", "bool"),
        ],
        "EventVariables": [
            ("HLT_IsoMu24", "bool"),
            ('MET_pt', 'float32'),
            ('MET_phi', 'float32'),
            ('MET_sumet', 'float32'),
            ('MET_significance', 'float32'),
            ('MET_CovXX', 'float32'),
            ('MET_CovXY', 'float32'),
            ('MET_CovYY', 'float32'),
            ('PV_npvs', 'int32'),
        ]
    }
    if ismc:
        datastructures["Muon"] += [("Muon_genPartIdx", "int32")]
        datastructures["Electron"] += [("Electron_genPartIdx", "int32")]
        
        datastructures["GenPart"] = [
            ('GenPart_pt', 'float32'),
            ('GenPart_eta', 'float32'),
            ('GenPart_phi', 'float32'),
            ('GenPart_mass', 'float32'),
            ('GenPart_pdgId', 'int32'),
            ('GenPart_status', 'int32'),
        ]

    return datastructures

def get_selected_muons(muons, pt_cut_leading, pt_cut_subleading, aeta_cut, iso_cut):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha

    passes_iso = muons.pfRelIso03_all > iso_cut
    passes_id = muons.tightId == True
    passes_subleading_pt = muons.pt > pt_cut_subleading
    passes_leading_pt = muons.pt > pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < aeta_cut
    
    selected_muons =  (
        passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )
    
    selected_muons_leading = selected_muons & passes_leading_pt
    
    evs_all = NUMPY_LIB.ones(muons.numevents(), dtype=bool)

    #select events with at least 1 good muon
    evs_1mu = ha.sum_in_offsets(
        muons, selected_muons_leading, evs_all, muons.masks["all"]
    ) >= 1
    
    return selected_muons, evs_1mu

def get_selected_electrons(electrons, pt_cut_leading, pt_cut_subleading, aeta_cut, iso_cut):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha

    passes_iso = electrons.pfRelIso03_all > iso_cut
    passes_id = electrons.pfId == True
    passes_subleading_pt = electrons.pt > pt_cut_subleading
    passes_leading_pt = electrons.pt > pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(electrons.eta) < aeta_cut
    
    evs_all = NUMPY_LIB.ones(electrons.numevents(), dtype=bool)
    els_all = NUMPY_LIB.ones(electrons.numobjects(), dtype=bool)
    
    selected_electrons =  (
        passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )
    
    selected_electrons_leading = selected_electrons & passes_leading_pt
    
    ev_1el = ha.sum_in_offsets(
        electrons, selected_electrons_leading, evs_all, electrons.masks["all"]
    ) >= 1
        
    return selected_electrons, ev_1el

def apply_lepton_corrections(leptons, mask_leptons, lepton_weights):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha
    
    corrs = NUMPY_LIB.zeros_like(leptons.pt)
    ha.get_bin_contents(leptons.pt, lepton_weights[:, 0], lepton_weights[:-1, 1], corrs)
    
    #multiply the per-lepton weights for each event
    all_events = NUMPY_LIB.ones(leptons.numevents(), dtype=NUMPY_LIB.bool)
    corr_per_event = ha.prod_in_offsets(
        leptons, corrs, all_events, mask_leptons
    )
    
    return corr_per_event

def apply_jec(jets_pt_orig, bins, jecs):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha

    corrs = NUMPY_LIB.zeros_like(jets_pt_orig)
    ha.get_bin_contents(jets_pt_orig, bins, jecs, corrs)
    return 1.0 + corrs

def select_jets(jets, mu, el, selected_muons, selected_electrons, pt_cut, aeta_cut, jet_lepton_dr_cut, btag_cut):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha

    passes_id = jets.puId == True
    passes_aeta = NUMPY_LIB.abs(jets.eta) < aeta_cut
    passes_pt = jets.pt > pt_cut
    passes_btag = jets.btag > btag_cut
    
    selected_jets = passes_id & passes_aeta & passes_pt

    jets_pass_dr_mu = ha.mask_deltar_first(
        jets, selected_jets, mu,
        selected_muons, jet_lepton_dr_cut)
        
    jets_pass_dr_el = ha.mask_deltar_first(
        jets, selected_jets, el,
        selected_electrons, jet_lepton_dr_cut)
    
    selected_jets_no_lepton = selected_jets & jets_pass_dr_mu & jets_pass_dr_el
    selected_jets_btag = selected_jets_no_lepton & passes_btag
    
    return selected_jets_no_lepton, selected_jets_btag

def fill_histograms_several(hists, systematic_name, histname_prefix, variables, mask, weights, use_cuda):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha

    all_arrays = []
    all_bins = []
    num_histograms = len(variables)

    for array, varname, bins in variables:
        if (len(array) != len(variables[0][0]) or
            len(array) != len(mask) or
            len(array) != len(weights["nominal"])):
            raise Exception("Data array {0} is of incompatible size".format(varname))
        all_arrays += [array]
        all_bins += [bins]

    max_bins = max([b.shape[0] for b in all_bins])
    stacked_array = NUMPY_LIB.stack(all_arrays, axis=0)
    stacked_bins = np.concatenate(all_bins)
    nbins = np.array([len(b) for b in all_bins])
    nbins_sum = np.cumsum(nbins)
    nbins_sum = np.insert(nbins_sum, 0, [0])

    for weight_name, weight_array in weights.items():
        if use_cuda:
            nblocks = 32
            out_w = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several[nblocks, 1024](
                stacked_array, weight_array, mask, stacked_bins,
                NUMPY_LIB.array(nbins), NUMPY_LIB.array(nbins_sum), out_w, out_w2
            )
            cuda.synchronize()

            out_w = out_w.sum(axis=1)
            out_w2 = out_w2.sum(axis=1)

            out_w = NUMPY_LIB.asnumpy(out_w)
            out_w2 = NUMPY_LIB.asnumpy(out_w2)
        else:
            out_w = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several(
                stacked_array, weight_array, mask, stacked_bins,
                nbins, nbins_sum, out_w, out_w2
            )

        out_w_separated = [out_w[i, 0:nbins[i]-1] for i in range(num_histograms)]
        out_w2_separated = [out_w2[i, 0:nbins[i]-1] for i in range(num_histograms)]

        for ihist in range(num_histograms):
            hist_name = histname_prefix + variables[ihist][1]
            bins = variables[ihist][2]
            target_histogram = Histogram(out_w_separated[ihist], out_w2_separated[ihist], bins)
            target = {weight_name: target_histogram}
            update_histograms_systematic(hists, hist_name, systematic_name, target)

def update_histograms_systematic(hists, hist_name, systematic_name, target_histogram):

    if hist_name not in hists:
        hists[hist_name] = {}

    if systematic_name[0] == "nominal" or systematic_name == "nominal":
        hists[hist_name].update(target_histogram)
    else:
        if systematic_name[1] == "":
            syst_string = systematic_name[0]
        else:
            syst_string = systematic_name[0] + "__" + systematic_name[1]
        target_histogram = {syst_string: target_histogram["nominal"]}
        hists[hist_name].update(target_histogram)

def run_analysis(dataset, out, dnnmodel, use_cuda, ismc):
    this_worker = get_worker()
    NUMPY_LIB = this_worker.NUMPY_LIB
    ha = this_worker.ha
    hists = {}
    histo_bins = {
        "nmu": np.array([0,1,2,3], dtype=np.float32),
        "njet": np.array([0,1,2,3,4,5,6,7], dtype=np.float32),
        "mu_pt": np.linspace(0, 300, 20),
        "mu_eta": np.linspace(-5, 5, 20),
        "mu_phi": np.linspace(-5, 5, 20),
        "mu_iso": np.linspace(0, 1, 20),
        "mu_charge": np.array([-1, 0, 1], dtype=np.float32),
        "met_pt": np.linspace(0,200,20),
        "jet_pt": np.linspace(0,400,20),
        "jet_eta": np.linspace(-5,5,20),
        "jet_phi": np.linspace(-5,5,20),
        "jet_btag": np.linspace(0,1,20),
        "dnnpred_m": np.linspace(0,1,20),
        "dnnpred_s": np.linspace(0,0.2,20),
        "inv_mass": np.linspace(150,200, 20),
        "sumpt": np.linspace(0,1000,20),
    }

    t0 = time.time()
 
    i = 0
 
    mu = dataset.structs["Muon"][i]
    el = dataset.structs["Electron"][i]
    jets = dataset.structs["Jet"][i]
    evvars = dataset.eventvars[i]

    mu.hepaccelerate_backend = ha
    el.hepaccelerate_backend = ha
    jets.hepaccelerate_backend = ha
    
    evs_all = NUMPY_LIB.ones(dataset.numevents(), dtype=NUMPY_LIB.bool)

    print("Lepton selection")
    sel_mu, sel_ev_mu = get_selected_muons(mu, 40, 20, 2.4, 0.1)
    sel_ev_mu = sel_ev_mu & (evvars['HLT_IsoMu24'] == True)
    mu.masks["selected"] = sel_mu
    sel_el, sel_ev_el = get_selected_electrons(el, 40, 20, 2.4, 0.1)
    el.masks["selected"] = sel_el
    
    nmu = ha.sum_in_offsets(
        mu, mu.masks["selected"], evs_all, mu.masks["all"], dtype=NUMPY_LIB.int32
    )
    nel = ha.sum_in_offsets(
        el, el.masks["selected"], evs_all, el.masks["all"], dtype=NUMPY_LIB.int32
    )
        
    #get contiguous arrays of the first two muons for all events
    mu1 = mu.select_nth(0, object_mask=sel_mu)
    mu2 = mu.select_nth(1, object_mask=sel_mu)
    el1 = el.select_nth(0, object_mask=sel_el)
    el2 = el.select_nth(1, object_mask=sel_el)
    
    weight_ev_mu = apply_lepton_corrections(mu, sel_mu, this_worker.electron_weights)
    weight_ev_el = apply_lepton_corrections(el, sel_el, this_worker.electron_weights)
   
    weights = {"nominal": weight_ev_mu * weight_ev_el}

    weights_jet = {}
    for k in weights.keys():
        weights_jet[k] = NUMPY_LIB.zeros_like(jets.pt)
        ha.broadcast(weights["nominal"], jets.offsets, weights_jet[k])

    all_jecs = [("nominal", "", None)]
    if ismc:
        for i in range(this_worker.jecs_up.shape[1]):
            all_jecs += [(i, "up", this_worker.jecs_up[:, i])]
            all_jecs += [(i, "down", this_worker.jecs_down[:, i])]
    
    jets_pt_orig = NUMPY_LIB.copy(jets.pt)

    #per-event histograms
    fill_histograms_several(
        hists, "nominal", "hist__all__",
        [
            (evvars["MET_pt"], "met_pt", histo_bins["met_pt"]),
        ],
        evs_all,
        weights,
        use_cuda,
    )

    fill_histograms_several(
        hists, "nominal", "hist__all__",
        [
            (jets.pt, "jets_pt", histo_bins["jet_pt"]),
        ],
        jets.masks["all"],
        weights_jet,
        use_cuda,
    )

    print("Jet selection")
    #loop over the jet corrections
    for ijec, sdir, jec in all_jecs:
        systname = "nominal"
        if ijec != "nominal":
            systname = ("jec{0}".format(ijec), sdir)
 
        if not jec is None:
            jet_pt_corr = apply_jec(jets_pt_orig, this_worker.jecs_bins, jec)
            #compute the corrected jet pt        
            jets.pt = jets_pt_orig * NUMPY_LIB.abs(jet_pt_corr)
        print("jec", ijec, sdir, jets.pt.mean())

        #get selected jets
        sel_jet, sel_bjet = select_jets(jets, mu, el, sel_mu, sel_el, 40, 2.0, 0.3, 0.4)
        
        #compute the number of jets per event 
        njet = ha.sum_in_offsets(
            jets, sel_jet, evs_all, jets.masks["all"], dtype=NUMPY_LIB.int32
        )
        nbjet = ha.sum_in_offsets(
            jets, sel_bjet, evs_all, jets.masks["all"], dtype=NUMPY_LIB.int32
        )

        inv_mass_3j = NUMPY_LIB.zeros(jets.numevents(), dtype=NUMPY_LIB.float32)
        best_comb_3j = NUMPY_LIB.zeros((jets.numevents(), 3), dtype=NUMPY_LIB.int32)

        if use_cuda:
            this_worker.kernels.comb_3_invmass_closest[32,256](jets.pt, jets.eta, jets.phi, jets.mass, jets.offsets, 172.0, inv_mass_3j, best_comb_3j)
            cuda.synchronize()
        else:
            this_worker.kernels.comb_3_invmass_closest(jets.pt, jets.eta, jets.phi, jets.mass, jets.offsets, 172.0, inv_mass_3j, best_comb_3j)

        best_btag = NUMPY_LIB.zeros(jets.numevents(), dtype=NUMPY_LIB.float32)
        if use_cuda:
            this_worker.kernels.max_val_comb[32,1024](jets.btag, jets.offsets, best_comb_3j, best_btag)
            cuda.synchronize()
        else:
            this_worker.kernels.max_val_comb(jets.btag, jets.offsets, best_comb_3j, best_btag)

        #get the events with at least three jets
        sel_ev_jet = (njet >= 3)
        sel_ev_bjet = (nbjet >= 1)
        
        selected_events = (sel_ev_mu | sel_ev_el) & sel_ev_jet & sel_ev_bjet
        print("Selected {0} events".format(selected_events.sum()))

        #get contiguous vectors of the first two jet data
        jet1 = jets.select_nth(0, object_mask=sel_jet)
        jet2 = jets.select_nth(1, object_mask=sel_jet)
        jet3 = jets.select_nth(2, object_mask=sel_jet)
       
        #create a mask vector for the first two jets 
        first_two_jets = NUMPY_LIB.zeros_like(sel_jet)
        inds = NUMPY_LIB.zeros_like(evs_all, dtype=NUMPY_LIB.int32) 
        targets = NUMPY_LIB.ones_like(evs_all, dtype=NUMPY_LIB.int32) 
        inds[:] = 0
        ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, selected_events, sel_jet)
        inds[:] = 1
        ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, selected_events, sel_jet)

        #compute the invariant mass of the first two jets
        dijet_inv_mass, dijet_pt = compute_inv_mass(jets, selected_events, sel_jet & first_two_jets, use_cuda)

        sumpt_jets = ha.sum_in_offsets(jets, jets.pt, selected_events, sel_jet)

        #create a keras-like array
        arr = NUMPY_LIB.vstack([
            nmu, nel, njet, dijet_inv_mass, dijet_pt, 
            mu1["pt"], mu1["eta"], mu1["phi"], mu1["charge"], mu1["pfRelIso03_all"],
            mu2["pt"], mu2["eta"], mu2["phi"], mu2["charge"], mu2["pfRelIso03_all"],
            el1["pt"], el1["eta"], el1["phi"], el1["charge"], el1["pfRelIso03_all"],
            el2["pt"], el2["eta"], el2["phi"], el2["charge"], el2["pfRelIso03_all"],
            jet1["pt"], jet1["eta"], jet1["phi"], jet1["btag"],
            jet2["pt"], jet2["eta"], jet2["phi"], jet2["btag"],
            inv_mass_3j, best_btag, sumpt_jets
        ]).T
       
        print("evaluating DNN model") 
        #pred = dnnmodel.eval(arr, use_cuda)
        #pred = NUMPY_LIB.vstack(pred).T
        #pred_m = NUMPY_LIB.mean(pred, axis=1)
        #pred_s = NUMPY_LIB.std(pred, axis=1)

        fill_histograms_several(
            hists, systname, "hist__nmu1_njetge3_nbjetge1__",
            [
                #(pred_m, "pred_m", histo_bins["dnnpred_m"]),
                #(pred_s, "pred_s", histo_bins["dnnpred_s"]),
                (nmu, "nmu", histo_bins["nmu"]),
                (nel, "nel", histo_bins["nmu"]),
                (njet, "njet", histo_bins["njet"]),

                (mu1["pt"], "mu1_pt", histo_bins["mu_pt"]),
                (mu1["eta"], "mu1_eta", histo_bins["mu_eta"]),
                (mu1["phi"], "mu1_phi", histo_bins["mu_phi"]),
                (mu1["charge"], "mu1_charge", histo_bins["mu_charge"]),
                (mu1["pfRelIso03_all"], "mu1_iso", histo_bins["mu_iso"]),

                (mu2["pt"], "mu2_pt", histo_bins["mu_pt"]),
                (mu2["eta"], "mu2_eta", histo_bins["mu_eta"]),
                (mu2["phi"], "mu2_phi", histo_bins["mu_phi"]),
                (mu2["charge"], "mu2_charge", histo_bins["mu_charge"]),
                (mu2["pfRelIso03_all"], "mu2_iso", histo_bins["mu_iso"]),

                (el1["pt"], "el1_pt", histo_bins["mu_pt"]),
                (el1["eta"], "el1_eta", histo_bins["mu_eta"]),
                (el1["phi"], "el1_phi", histo_bins["mu_phi"]),
                (el1["charge"], "el1_charge", histo_bins["mu_charge"]),
                (el1["pfRelIso03_all"], "el1_iso", histo_bins["mu_iso"]),
                
                (el2["pt"], "el2_pt", histo_bins["mu_pt"]),
                (el2["eta"], "el2_eta", histo_bins["mu_eta"]),
                (el2["phi"], "el2_phi", histo_bins["mu_phi"]),
                (el2["charge"], "el2_charge", histo_bins["mu_charge"]),
                (el2["pfRelIso03_all"], "el2_iso", histo_bins["mu_iso"]),
                
                (jet1["pt"], "j1_pt", histo_bins["jet_pt"]),
                (jet1["eta"], "j1_eta", histo_bins["jet_eta"]),
                (jet1["phi"], "j1_phi", histo_bins["jet_phi"]),
                (jet1["btag"], "j1_btag", histo_bins["jet_btag"]),
                
                (jet2["pt"], "j2_pt", histo_bins["jet_pt"]),
                (jet2["eta"], "j2_eta", histo_bins["jet_eta"]),
                (jet2["phi"], "j2_phi", histo_bins["jet_phi"]),
                (jet2["btag"], "j2_btag", histo_bins["jet_btag"]),
                
                (inv_mass_3j, "inv_mass_3j", histo_bins["inv_mass"]),
                (best_btag, "best_btag", histo_bins["jet_btag"]),
                (sumpt_jets, "sumpt", histo_bins["sumpt"]),
            ],
            selected_events,
            weights,
            use_cuda
        )

        #save the array for the first jet correction scenario only
        if save_arrays and ijec == 0:
            outfile_arr = "{0}_arrs.npy".format(out)
            print("Saving array with shape {0} to {1}".format(arr.shape, outfile_arr))
            with open(outfile_arr, "wb") as fi:
                np.save(fi, NUMPY_LIB.asnumpy(arr))

    t1 = time.time()

    res = Results({})
    for hn in hists.keys():
        hists[hn] = Results(hists[hn])
    res["hists"] = Results(hists)
    res["numevents"] = dataset.numevents()

    speed = dataset.numevents() / (t1 - t0) 
    print("run_analysis: {0:.2E} events in {1:.2f} seconds, speed {2:.2E} Hz".format(dataset.numevents(), t1 - t0, speed))
    return res

def load_dataset(datapath, cachepath, filenames, ismc, nthreads, skip_cache, do_skim, NUMPY_LIB, ha):
    ds = hepaccelerate.Dataset(
        "dataset",
        filenames,
        create_datastructure(ismc),
        datapath=datapath,
        cache_location=cachepath,
        treename="aod2nanoaod/Events",
    )
    
    cache_valid = ds.check_cache()
   
    timing_results = {}
 
    if skip_cache or not cache_valid:
        
        #Load the ROOT files
        print("Loading dataset from {0} files".format(len(ds.filenames)))
        t0 = time.time()
        ds.preload(nthreads=nthreads)
        t1 = time.time()
        timing_results["load_root"] = t1 - t0

        ds.make_objects()
        ds.cache_metadata = ds.create_cache_metadata()
        print("Loaded dataset, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
        #Apply a skim on the trigger bit for each file
        if do_skim:
            masks = [v['HLT_IsoMu24']==True for v in ds.eventvars]
            ds.compact(masks)
            print("Applied trigger bit selection skim, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
        print("Saving skimmed data to uncompressed cache")
        t0 = time.time()
        ds.to_cache(nthreads=nthreads)
        t1 = time.time()
        timing_results["to_cache"] = t1 - t0
        speed = ds.numevents() / (t1 - t0)
        print("load_dataset: {0:.2E} events / second".format(speed))
    else:
        print("Loading from existing cache")
        t0 = time.time()
        ds.from_cache(verbose=True)
        t1 = time.time()
        timing_results["from_cache"] = t1 - t0
        speed = ds.numevents() / (t1 - t0)
        print("load_dataset: {0:.2E} events / second".format(speed))
    
    #Compute the average length of the data structures
    avg_vec_length = np.mean(np.array([[v["pt"].shape[0] for v in ds.structs[ss]] for ss in ds.structs.keys()]))
    print("Average vector length before merging: {0:.0f}".format(avg_vec_length))
  
    print("Merging arrays from multiple files using awkward") 
    #Now merge all the arrays across the files to have large contiguous data
    t0 = time.time()
    ds.merge_inplace()
    t1 = time.time()
    timing_results["merge_inplace"] = t1 - t0
    
    print("Copying to device")
    #transfer dataset to device (GPU) if applicable
    t0 = time.time()
    ds.move_to_device(NUMPY_LIB)
    t1 = time.time()
    timing_results["move_to_device"] = t1 - t0
    ds.numpy_lib = NUMPY_LIB

    avg_vec_length = np.mean(np.array([[v["pt"].shape[0] for v in ds.structs[ss]] for ss in ds.structs.keys()]))
    print("Average vector length after merging: {0:.0f}".format(avg_vec_length))

    return ds, timing_results

def compute_inv_mass(objects, mask_events, mask_objects, use_cuda):
    this_worker = get_worker()
    NUMPY_LIB, ha = this_worker.NUMPY_LIB, this_worker.ha 

    inv_mass = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    pt_total = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    if use_cuda:
        compute_inv_mass_cudakernel[32, 1024](
            objects.offsets, objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
        cuda.synchronize()
    else:
        compute_inv_mass_kernel(objects.offsets,
            objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
    return inv_mass, pt_total

@numba.njit(parallel=True, fastmath=True)
def compute_inv_mass_kernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    for iev in numba.prange(offsets.shape[0]-1):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = np.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = np.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

@cuda.jit
def compute_inv_mass_cudakernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for iev in range(xi, offsets.shape[0]-1, xstride):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px = pt * math.cos(phi)
                    py = pt * math.sin(phi)
                    pz = pt * math.sinh(eta)
                    e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = math.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = math.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis')
    parser.add_argument('--datapath', action='store',
        help='Input file path that contains the CMS /store/... folder, e.g. /mnt/hadoop',
        required=False, default="/storage/user/jpata")
    parser.add_argument('--cachepath', action='store',
        help='Location where to store the cache',
        required=False, default="./mycache")
    parser.add_argument('--ismc', action='store_true',
        help='Flag to specify if dataset is MC')
    parser.add_argument('--skim', action='store_true',
        help='Specify if skim should be done')
    parser.add_argument('--nocache', action='store_true',
        help='Flag to specify if branch cache will be skipped')
    parser.add_argument('--nthreads', action='store',
        help='Number of parallel threads', default=1, type=int)
    parser.add_argument('--out', action='store',
        help='Output file name', default="out.pkl")
    parser.add_argument('--njobs', action='store',
        help='Number of multiprocessing jobs', default=1, type=int)
    parser.add_argument('--njec', action='store',
        help='Number of JEC scenarios', default=20, type=int)

    parser.add_argument('filenames', nargs=argparse.REMAINDER)
 
    args = parser.parse_args()
    return args

def multiprocessing_initializer(args, use_cuda):
    try:
        this_worker = get_worker()
    except Exception as e:
        this_worker = None

    import tensorflow as tf
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads=args.nthreads
    config.inter_op_parallelism_threads=args.nthreads
    if not use_cuda: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        cpu_name = multiprocessing.current_process().name
        if cpu_name == "MainProcess":
            import setGPU
        else: 
            cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
            gpu_id = gpu_id_list[cpu_id]
            print("process {0} choosing GPU {1}".format(cpu_name, gpu_id))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        gpu_memory_fraction = 0.2
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    from keras.backend.tensorflow_backend import set_session
    set_session(tf.Session(config=config))
    this_worker.dnnmodel = DNNModel()
    
    NUMPY_LIB, ha = hepaccelerate.choose_backend(use_cuda)
    this_worker.NUMPY_LIB = NUMPY_LIB
    this_worker.ha = ha
    if use_cuda:
        import cuda_kernels as kernels
    else:
        import cpu_kernels as kernels
    this_worker.kernels = kernels

    #Create random vectors as placeholders of lepton pt event weights
    this_worker.electron_weights = NUMPY_LIB.zeros((100, 2), dtype=NUMPY_LIB.float32)
    this_worker.electron_weights[:, 0] = NUMPY_LIB.linspace(0, 200, this_worker.electron_weights.shape[0])[:]
    this_worker.electron_weights[:, 1] = 1.0

    #Create random vectors as placeholders of pt-dependent jet energy corrections
    this_worker.jecs_bins = NUMPY_LIB.zeros(100, dtype=NUMPY_LIB.float32 ) 
    this_worker.jecs_up = NUMPY_LIB.zeros((99, args.njec), dtype=NUMPY_LIB.float32)
    this_worker.jecs_down = NUMPY_LIB.zeros((99, args.njec), dtype=NUMPY_LIB.float32)
    this_worker.jecs_bins[:] = NUMPY_LIB.linspace(0, 200, this_worker.jecs_bins.shape[0])[:]

    for i in range(args.njec):
        this_worker.jecs_up[:, i] = 0.3*(float(i+1)/float(args.njec)) 
        this_worker.jecs_down[:, i] = -0.3*(float(i+1)/float(args.njec)) 

def load_and_analyze(args_tuple):
    fn, args, dataset, ismc, ichunk = args_tuple
    this_worker = get_worker()
    NUMPY_LIB, ha = hepaccelerate.choose_backend(args.use_cuda)
    
    print("Loading {0}".format(fn))
    ds, timing_results = load_dataset(args.datapath, args.cachepath, fn, ismc, args.nthreads, args.nocache, args.skim, NUMPY_LIB, ha)
    t0 = time.time()
    ret = run_analysis(ds, "{0}_{1}".format(dataset, ichunk), this_worker.dnnmodel, use_cuda, ismc)
    t1 = time.time()
    ret["timing"] = Results(timing_results)
    ret["timing"]["run_analysis"] = t1 - t0
    ret["timing"]["num_events"] = ds.numevents()
    return ret

parameters = {
    "muon_pt_leading": 40,
    "muon_pt": 20,
    "muon_eta": 2.4,
    "muon_iso": 0.3,
    "electron_pt_leading": 40,
    "electron_pt": 20,
    "electron_eta": 2.4,
    "electron_iso": 0.3,
    "jet_pt": 30,
    "jet_eta": 2.0,
    "jet_lepton_dr": 0.3,
    "jet_btag": 0.4
}

if __name__ == "__main__":
    np.random.seed(0)
    args = parse_args()
    args.use_cuda = use_cuda
    for i in range(1):
        download_if_not_exists("data/model_kf{0}.h5".format(i), "https://jpata.web.cern.ch/jpata/hepaccelerate/model_kf{0}.h5".format(i))

    from dask.distributed import Client, LocalCluster
    from distributed import get_worker

    cluster = LocalCluster(n_workers=args.njobs, threads_per_worker=args.nthreads, memory_limit=0)
    client = Client(cluster)

    #run initialization
    client.run(multiprocessing_initializer, args, use_cuda)
    
    print("Processing all datasets")
    datasets = [
        ("DYJetsToLL", "/opendata_files/DYJetsToLL-merged/*.root", True),
        ("TTJets_FullLeptMGDecays", "/opendata_files/TTJets_FullLeptMGDecays-merged/*.root", True),
        ("TTJets_Hadronic", "/opendata_files/TTJets_Hadronic-merged/*.root", True),
        ("TTJets_SemiLeptMGDecays", "/opendata_files/TTJets_SemiLeptMGDecays-merged/*.root", True),
        ("W1JetsToLNu", "/opendata_files/W1JetsToLNu-merged/*.root", True),
        ("W2JetsToLNu", "/opendata_files/W2JetsToLNu-merged/*.root", True),
        ("W3JetsToLNu", "/opendata_files/W3JetsToLNu-merged/*.root", True),
        ("GluGluToHToMM", "/opendata_files/GluGluToHToMM-merged/*.root", True),
        ("SingleMu", "/opendata_files/SingleMu-merged/*.root", False),
    ]
    arglist = []

    walltime_t0 = time.time()
    for dataset, fn_pattern, ismc in datasets:
        filenames = glob.glob(args.datapath + fn_pattern)
        if(len(filenames) == 0):
            raise Exception("Could not find any filenames for dataset={0}: {{datapath}}/{{fn_pattern}}={1}/{2}".format(dataset, args.datapath, fn_pattern))
        ichunk = 0
        for fn in chunks(filenames, 1):
            arglist += [(fn, args, dataset, ismc, ichunk)]
            ichunk += 1

    print("Processing {0} arguments".format(len(arglist)))
    futures = client.map(load_and_analyze, arglist)
    ret = [fut.result() for fut in futures]
    walltime_t1 = time.time()

    print("Merging outputs")
    hists = {ds[0]: [] for ds in datasets}
    numevents = {ds[0]: 0 for ds in datasets}
    for r, _args in zip(ret, arglist):
        rh = r["hists"]
        ds = _args[2]
        hists[ds] += [Results(r["hists"])]
        numevents[ds] += r["numevents"]

    timing = sum([r["timing"] for r in ret], Results({}))
    timing["cuda"] = use_cuda
    timing["njec"] = args.njec
    timing["nthreads"] = args.nthreads
    timing["walltime"] = walltime_t1 - walltime_t0

    for k, v in hists.items():
        hists[k] = sum(hists[k], Results({}))
    
    print("Writing output pkl")
    with open(args.out, "wb") as fi:
        pickle.dump({"hists": hists, "numevents": numevents, "timing": timing}, fi)
    client.shutdown()
    sum_numev = sum(numevents.values())
    print("Processed {0} events in {1:.1f} seconds, {2:.2E} Hz".format(sum_numev, timing["walltime"], sum_numev / timing["walltime"]))
