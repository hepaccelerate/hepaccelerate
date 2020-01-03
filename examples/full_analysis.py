import os, glob, sys, time, argparse, multiprocessing
import pickle, math, requests
from collections import OrderedDict

import distributed
from distributed import WorkerPlugin

import uproot
import hepaccelerate
import hepaccelerate.kernels as ha_kernels
from hepaccelerate.utils import Histogram, Results

import numba
from numba import cuda
import numpy as np

import dask
from dask.distributed import Client
from distributed import get_worker

use_cuda = bool(int(os.environ.get("HEPACCELERATE_CUDA", 0)))

#save training arrays for DNN
save_arrays = False

#Run once on each worker to load various configuration info and initialize tensorflow correctly
class InitializerPlugin(WorkerPlugin):
    def __init__(self, args):
        self.args = args
        pass

    def setup(self, worker: distributed.Worker):
        multiprocessing_initializer(self.args)

    def teardown(self, worker: distributed.Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        pass

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
    def __init__(self, NUMPY_LIB):
        import keras
        self.models = []
        for i in range(2):
            self.models += [keras.models.load_model("data/model_kf{0}.h5".format(i))]
        self.models[0].summary()
        self.NUMPY_LIB = NUMPY_LIB

    def eval(self, X, use_cuda):
        if use_cuda:
            X = self.NUMPY_LIB.asnumpy(X)
        rets = [self.NUMPY_LIB.array(m.predict(X, batch_size=10000)[:, 0]) for m in self.models]
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
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend

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
    evs_1mu = ha_kernels.sum_in_offsets(
        this_worker.backend,
        muons.offsets, selected_muons_leading, evs_all, muons.masks["all"]
    ) >= 1
    
    return selected_muons, evs_1mu

def get_selected_electrons(electrons, pt_cut_leading, pt_cut_subleading, aeta_cut, iso_cut):
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend

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
    
    ev_1el = ha_kernels.sum_in_offsets(
        this_worker.backend,
        electrons.offsets, selected_electrons_leading, evs_all, electrons.masks["all"]
    ) >= 1
        
    return selected_electrons, ev_1el

def apply_lepton_corrections(leptons, mask_leptons, lepton_weights):
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend
    
    corrs = NUMPY_LIB.zeros_like(leptons.pt)
    ha_kernels.get_bin_contents(backend, leptons.pt, lepton_weights[:, 0], lepton_weights[:-1, 1], corrs)
    
    #multiply the per-lepton weights for each event
    all_events = NUMPY_LIB.ones(leptons.numevents(), dtype=NUMPY_LIB.bool)
    corr_per_event = ha_kernels.prod_in_offsets(
        backend, 
        leptons.offsets, corrs, all_events, mask_leptons
    )
    
    return corr_per_event

def apply_jec(jets_pt_orig, bins, jecs):
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend

    corrs = NUMPY_LIB.zeros_like(jets_pt_orig)
    ha_kernels.get_bin_contents(backend, jets_pt_orig, bins, jecs, corrs)
    return 1.0 + corrs

def select_jets(jets, mu, el, selected_muons, selected_electrons, pt_cut, aeta_cut, jet_lepton_dr_cut, btag_cut):
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend

    passes_id = jets.puId == True
    passes_aeta = NUMPY_LIB.abs(jets.eta) < aeta_cut
    passes_pt = jets.pt > pt_cut
    passes_btag = jets.btag > btag_cut
    
    selected_jets = passes_id & passes_aeta & passes_pt

    jets_d = {"eta": jets.eta, "phi": jets.phi, "offsets": jets.offsets} 
    jets_pass_dr_mu = ha_kernels.mask_deltar_first(
        backend,
        jets_d,
        selected_jets,
        {"eta": mu.eta, "phi": mu.phi, "offsets": mu.offsets},
        selected_muons, jet_lepton_dr_cut)
        
    jets_pass_dr_el = ha_kernels.mask_deltar_first(
        backend,
        jets_d,
        selected_jets,
        {"eta": el.eta, "phi": el.phi, "offsets": el.offsets},
        selected_electrons, jet_lepton_dr_cut)
    
    selected_jets_no_lepton = selected_jets & jets_pass_dr_mu & jets_pass_dr_el
    selected_jets_btag = selected_jets_no_lepton & passes_btag
    
    return selected_jets_no_lepton, selected_jets_btag

def fill_histograms_several(hists, systematic_name, histname_prefix, variables, mask, weights, use_cuda):
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend

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
            backend.fill_histogram_several[nblocks, 1024](
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
            backend.fill_histogram_several(
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
    from keras.backend.tensorflow_backend import set_session
    this_worker = get_worker_wrapper()
    NUMPY_LIB = this_worker.NUMPY_LIB
    backend = this_worker.backend
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

    mu.hepaccelerate_backend = backend
    el.hepaccelerate_backend = backend 
    jets.hepaccelerate_backend = backend 
    
    evs_all = NUMPY_LIB.ones(dataset.numevents(), dtype=NUMPY_LIB.bool)

    print("Lepton selection")
    sel_mu, sel_ev_mu = get_selected_muons(mu, 40, 20, 2.4, 0.1)
    sel_ev_mu = sel_ev_mu & (evvars['HLT_IsoMu24'] == True)
    mu.masks["selected"] = sel_mu
    sel_el, sel_ev_el = get_selected_electrons(el, 40, 20, 2.4, 0.1)
    el.masks["selected"] = sel_el
    
    nmu = ha_kernels.sum_in_offsets(
        backend,
        mu.offsets, mu.masks["selected"], evs_all, mu.masks["all"], dtype=NUMPY_LIB.int32
    )
    nel = ha_kernels.sum_in_offsets(
        backend,
        el.offsets, el.masks["selected"], evs_all, el.masks["all"], dtype=NUMPY_LIB.int32
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
        ha_kernels.broadcast(backend, jets.offsets, weights["nominal"], weights_jet[k])

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
        njet = ha_kernels.sum_in_offsets(
            backend,
            jets.offsets, sel_jet, evs_all, jets.masks["all"], dtype=NUMPY_LIB.int32
        )
        nbjet = ha_kernels.sum_in_offsets(
            backend,
            jets.offsets, sel_bjet, evs_all, jets.masks["all"], dtype=NUMPY_LIB.int32
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
        ha_kernels.set_in_offsets(backend, jets.offsets, first_two_jets, inds, targets, selected_events, sel_jet)
        inds[:] = 1
        ha_kernels.set_in_offsets(backend, jets.offsets, first_two_jets, inds, targets, selected_events, sel_jet)

        #compute the invariant mass of the first two jets
        dijet_inv_mass, dijet_pt = compute_inv_mass(jets, selected_events, sel_jet & first_two_jets, use_cuda)

        sumpt_jets = ha_kernels.sum_in_offsets(backend, jets.offsets, jets.pt, selected_events, sel_jet)

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
       
        #print("evaluating DNN model") 
        with this_worker.graph.as_default():
            set_session(this_worker.session) 
            pred = dnnmodel.eval(arr, use_cuda)
            pred = NUMPY_LIB.vstack(pred).T
            pred_m = NUMPY_LIB.mean(pred, axis=1)
            pred_s = NUMPY_LIB.std(pred, axis=1)

        fill_histograms_several(
            hists, systname, "hist__nmu1_njetge3_nbjetge1__",
            [
                (pred_m, "pred_m", histo_bins["dnnpred_m"]),
                (pred_s, "pred_s", histo_bins["dnnpred_s"]),
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

def load_dataset(datapath, filenames, ismc, nthreads, do_skim, NUMPY_LIB, ha, entrystart, entrystop):
    ds = hepaccelerate.Dataset(
        "dataset",
        filenames,
        create_datastructure(ismc),
        datapath=datapath,
        treename="Events",
    )
    
    timing_results = {}
 
    #Load the ROOT files
    print("Loading dataset from {0} files".format(len(ds.filenames)))
    t0 = time.time()
    ds.preload(nthreads=nthreads, entrystart=entrystart, entrystop=entrystop)
    t1 = time.time()
    ds.make_objects()
    timing_results["load_root"] = t1 - t0
    print("Loaded dataset, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
    #Apply a skim on the trigger bit for each file
    if do_skim:
        masks = [v['HLT_IsoMu24']==True for v in ds.eventvars]
        ds.compact(masks)
        print("Applied trigger bit selection skim, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
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
    this_worker = get_worker_wrapper()
    NUMPY_LIB, backend = this_worker.NUMPY_LIB, this_worker.backend 

    inv_mass = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    pt_total = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    if use_cuda:
        this_worker.kernels.compute_inv_mass_cudakernel[32, 1024](
            objects.offsets, objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
        cuda.synchronize()
    else:
        this_worker.kernels.compute_inv_mass_kernel(objects.offsets,
            objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
    return inv_mass, pt_total

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis')
    parser.add_argument('--datapath', action='store',
        help='Input file path that contains the CMS /store/... folder, e.g. /mnt/hadoop',
        required=False, default=".")
    parser.add_argument('--dask-server', action='store',
        help='IP of the dask server',
        required=False, default="127.0.0.1:8786")
    parser.add_argument('--skim', action='store_true',
        help='Specify if skim should be done')
    parser.add_argument('--nthreads', action='store',
        help='Number of parallel threads', default=1, type=int)
    parser.add_argument('--out', action='store',
        help='Output file name', default="out.pkl")
    parser.add_argument('--njobs', action='store',
        help='Number of multiprocessing jobs', default=1, type=int)
    parser.add_argument('--njec', action='store',
        help='Number of JEC scenarios', default=1, type=int)
 
    args = parser.parse_args()
    
    #Will start a local cluster
    if args.dask_server == "":
        args.dask_server = None

    return args

#Placeholder module for debugging without dask
class Module:
    pass
global_worker = Module()

#Get either the dask worker or the global module
def get_worker_wrapper():
    global global_worker
    try:
        this_worker = get_worker()
    except Exception as e:
        this_worker = global_worker
    return this_worker

#Initialize the worker: load tensorflow and kernels
def multiprocessing_initializer(args, gpu_id=None):
    this_worker = get_worker_wrapper()

    #Set up tensorflow
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads=args.nthreads
    config.inter_op_parallelism_threads=args.nthreads
    os.environ["NUMBA_NUM_THREADS"] = str(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
    if not args.use_cuda: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        if not gpu_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config.gpu_options.allow_growth = False
        gpu_memory_fraction = 0.2
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

    from keras.backend.tensorflow_backend import set_session
    this_worker.session = tf.Session(config=config)
    set_session(this_worker.session)
    
    NUMPY_LIB, backend = hepaccelerate.choose_backend(args.use_cuda)
    this_worker.dnnmodel = DNNModel(NUMPY_LIB)
    this_worker.NUMPY_LIB = NUMPY_LIB
    this_worker.backend = backend

    #Import kernels that are specific to this analysis 
    if args.use_cuda:
        import cuda_kernels as kernels
    else:
        import cpu_kernels as kernels

    this_worker.kernels = kernels
    this_worker.graph = tf.get_default_graph()

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
    fn, args, dataset, entrystart, entrystop, ismc, ichunk = args_tuple
    this_worker = get_worker_wrapper()
    NUMPY_LIB, backend = hepaccelerate.choose_backend(args.use_cuda)
    
    print("Loading {0}".format(fn))
    ds, timing_results = load_dataset(args.datapath, fn, ismc, args.nthreads, args.skim, NUMPY_LIB, backend, entrystart, entrystop)
    t0 = time.time()
    ret = run_analysis(ds, "{0}_{1}".format(dataset, ichunk), this_worker.dnnmodel, args.use_cuda, ismc)
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

if __name__ == "__main__":
    np.random.seed(0)
    args = parse_args()
    args.use_cuda = use_cuda
    for i in range(2):
        download_if_not_exists("data/model_kf{0}.h5".format(i), "https://jpata.web.cern.ch/jpata/hepaccelerate/model_kf{0}.h5".format(i))

    print("Trying to connect to dask cluster, please start it with examples/dask_cluster.sh or examples/dask_cluster_gpu.sh")

    if args.dask_server == "debug":
        multiprocessing_initializer(args)
    else:
        client = Client(args.dask_server)
        plugin = InitializerPlugin(args)
        client.register_worker_plugin(plugin)
 
    print("Processing all datasets")
    arglist = []

    walltime_t0 = time.time()
    for dataset, fn_pattern, ismc in datasets:
        filenames = glob.glob(args.datapath + fn_pattern)
        if(len(filenames) == 0):
            raise Exception("Could not find any filenames for dataset={0}: {{datapath}}/{{fn_pattern}}={1}/{2}".format(dataset, args.datapath, fn_pattern))
        ichunk = 0
        for fn in filenames:
            nev = len(uproot.open(fn).get("Events"))
            #Process in chunks of 500k events to limit peak memory usage
            for evs_chunk in chunks(range(nev), 500000):
                entrystart = evs_chunk[0] 
                entrystop = evs_chunk[-1] + 1
                arglist += [([fn], args, dataset, entrystart, entrystop, ismc, ichunk)]
                ichunk += 1

    print("Processing {0} arguments".format(len(arglist)))

    if args.dask_server == "debug":
        ret = map(load_and_analyze, arglist)
    else:
        futures = client.map(load_and_analyze, arglist, retries=3)
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
