import time
import os
import numba
import requests
import unittest
import numpy as np
import json

from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import uproot



use_cuda = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
def download_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)

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

class TestKernels(unittest.TestCase):
    def setUp(self):
        self.use_cuda = use_cuda
        self.NUMPY_LIB, self.ha = choose_backend(use_cuda=self.use_cuda)
        self.dataset = TestKernels.load_dataset()

    @staticmethod
    def load_dataset():
        download_if_not_exists("data/nanoaod_test.root", "https://jpata.web.cern.ch/jpata/nanoaod_test.root")
        datastructures = {
            "Muon": [
                ("Muon_pt", "float32"), ("Muon_eta", "float32"),
                ("Muon_phi", "float32"), ("Muon_mass", "float32"),
                ("Muon_pdgId", "int32"),
                ("Muon_pfRelIso04_all", "float32"), ("Muon_mediumId", "bool"),
                ("Muon_tightId", "bool"), ("Muon_charge", "int32"),
                ("Muon_isGlobal", "bool"), ("Muon_isTracker", "bool"),
                ("Muon_nTrackerLayers", "int32"),
            ],
            "Electron": [
                ("Electron_pt", "float32"), ("Electron_eta", "float32"),
                ("Electron_phi", "float32"), ("Electron_mass", "float32"),
                ("Electron_mvaFall17V1Iso_WP90", "bool"),
            ],
            "Jet": [
                ("Jet_pt", "float32"), ("Jet_eta", "float32"),
                ("Jet_phi", "float32"), ("Jet_mass", "float32"),
                ("Jet_btagDeepB", "float32"),
                ("Jet_jetId", "int32"), ("Jet_puId", "int32"),
            ],
            "TrigObj": [
                ("TrigObj_pt", "float32"),
                ("TrigObj_eta", "float32"),
                ("TrigObj_phi", "float32"),
                ("TrigObj_id", "int32")
            ],
            "EventVariables": [
                ("PV_npvsGood", "float32"), 
                ("PV_ndof", "float32"),
                ("PV_z", "float32"),
                ("Flag_BadChargedCandidateFilter", "bool"),
                ("Flag_HBHENoiseFilter", "bool"),
                ("Flag_HBHENoiseIsoFilter", "bool"),
                ("Flag_EcalDeadCellTriggerPrimitiveFilter", "bool"),
                ("Flag_goodVertices", "bool"),
                ("Flag_globalSuperTightHalo2016Filter", "bool"),
                ("Flag_BadPFMuonFilter", "bool"),
                ("Flag_BadChargedCandidateFilter", "bool"),
                ("HLT_IsoMu27", "bool"),
                ("run", "uint32"),
                ("luminosityBlock", "uint32"),
                ("event", "uint64")
            ],
        }
        datastructures["EventVariables"] += [
            ("Pileup_nTrueInt", "uint32"),
            ("Generator_weight", "float32"),
            ("genWeight", "float32")
        ]
        datastructures["Muon"] += [
            ("Muon_genPartIdx", "int32"),
        ]
        datastructures["GenPart"] = [
            ("GenPart_pt", "float32"),
        ]
        dataset = Dataset("nanoaod", ["data/nanoaod_test.root"], datastructures, cache_location="./mycache/", treename="Events", datapath="")
      
        try:
            dataset.from_cache()
        except Exception as e:
            dataset.load_root()
            dataset.to_cache()
    
        return dataset

    def time_kernel(self, test_kernel):
        test_kernel()
    
        t0 = time.time()
        for i in range(5):
            n = test_kernel()
        t1 = time.time()
    
        dt = (t1 - t0) / 5.0
        speed = float(n)/dt
        return speed

    
    def test_kernel_sum_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        z = self.ha.sum_in_offsets(
            muons,
            muons.pt,
            sel_ev,
            sel_mu, dtype=self.NUMPY_LIB.float32)
        return muons.numevents()

    def test_kernel_max_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        z = self.ha.max_in_offsets(
            muons,
            muons.pt,
            sel_ev,
            sel_mu)
        return muons.numevents()

    def test_kernel_simple_cut(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_mu = muons.pt > 30.0
        return muons.numevents()

    def test_kernel_mask_deltar_first(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        trigobj = dataset.structs["TrigObj"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        sel_trigobj = (trigobj.id == 13)
        muons_matched_to_trigobj = self.ha.mask_deltar_first(
            muons, sel_mu, trigobj,
            sel_trigobj, 0.5
        )
        return muons.numevents()
        
    def test_kernel_select_muons_opposite_sign(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        muons_passing_os = self.ha.select_muons_opposite_sign(
                muons, sel_mu)
        return muons.numevents()
    
    def test_kernel_histogram_from_vector(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        weights = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.float32)
        self.ha.histogram_from_vector(muons.pt, weights, self.NUMPY_LIB.linspace(0,200,100))
        return muons.numevents()

    def test_timing(self):
        with open("kernel_benchmarks.txt", "a") as of:
            for i in range(5):
                ret = self.run_timing()
                of.write(json.dumps(ret) + '\n')

    def run_timing(self):
        ds = self.dataset

        t0 = time.time()
        for i in range(5):
            ds.move_to_device(self.NUMPY_LIB)
        t1 = time.time()
        dt = (t1 - t0)/5.0

        ret = {"use_cuda": self.use_cuda, "num_threads": numba.config.NUMBA_NUM_THREADS, "use_avx": numba.config.ENABLE_AVX}
        print("Memory transfer speed: {0:.2f} MHz, event size {1:.2f} bytes, data transfer speed {2:.2f} MB/s".format(ds.numevents() / dt / 1000.0 / 1000.0, ds.eventsize(), ds.memsize()/dt/1000/1000))
        ret["memory_transfer"] = ds.numevents() / dt / 1000.0 / 1000.0

        t = self.time_kernel(self.test_kernel_sum_in_offsets)
        print("sum_in_offsets {0:.2f} MHz".format(t/1000/1000))
        ret["sum_in_offsets"] = t/1000/1000
    
        t = self.time_kernel(self.test_kernel_simple_cut)
        print("simple_cut {0:.2f} MHz".format(t/1000/1000))
        ret["simple_cut"] = t/1000/1000
    
        t = self.time_kernel(self.test_kernel_max_in_offsets)
        print("max_in_offsets {0:.2f} MHz".format(t/1000/1000))
        ret["max_in_offsets"] = t/1000/1000

        t = self.time_kernel(self.test_kernel_mask_deltar_first)
        print("mask_deltar_first {0:.2f} MHz".format(t/1000/1000))
        ret["mask_deltar_first"] = t/1000/1000
        
        t = self.time_kernel(self.test_kernel_select_muons_opposite_sign)
        print("select_muons_opposite_sign {0:.2f} MHz".format(t/1000/1000))
        ret["select_muons_opposite_sign"] = t/1000/1000
        
        t = self.time_kernel(self.test_kernel_histogram_from_vector)
        print("histogram_from_vector {0:.2f} MHz".format(t/1000/1000))
        ret["histogram_from_vector"] = t/1000/1000
        
        return ret 

if __name__ == "__main__":
    unittest.main()
