import time
import os
import numba
import requests
import unittest
import numpy as np
import json
import sys

import hepaccelerate
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend
import uproot

USE_CUDA = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
    
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
    NUMPY_LIB, ha = choose_backend(use_cuda=USE_CUDA)
    use_cuda = USE_CUDA
    
    def setUp(self):
        self.dataset = dataset

    @staticmethod
    def load_dataset(numpy_lib):
        print("loading dataset")
        download_if_not_exists("data/nanoaod_test.root", "https://jpata.web.cern.ch/jpata/opendata_files/DY2JetsToLL-merged/1.root")
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
            ]
        }
        dataset = Dataset("nanoaod", ["./data/nanoaod_test.root"], datastructures, cache_location="./mycache/", treename="aod2nanoaod/Events", datapath="")
      
        try:
            dataset.from_cache()
        except Exception as e:
            dataset.load_root()
            dataset.to_cache()
        print("merging dataset")
        dataset.merge_inplace()
        print("dataset has {0} events, {1:.2f} MB".format(dataset.numevents(), dataset.memsize()/1000/1000))
        print("moving to device")
        dataset.move_to_device(numpy_lib)
 
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

    def test_kernel_get_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        inds = self.NUMPY_LIB.zeros(muons.numevents(), dtype=self.NUMPY_LIB.int8)
        inds[:] = 0
        z = self.ha.get_in_offsets(
            muons.pt,
            muons.offsets,
            inds,
            sel_ev,
            sel_mu)
        return muons.numevents()

    def test_kernel_set_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        inds = self.NUMPY_LIB.zeros(muons.numevents(), dtype=self.NUMPY_LIB.int8)
        inds[:] = 0
        target = self.NUMPY_LIB.ones(muons.numevents(), dtype=muons.pt.dtype)
        self.ha.set_in_offsets(
            muons.pt,
            muons.offsets,
            inds,
            target,
            sel_ev,
            sel_mu)

        z = self.ha.get_in_offsets(
            muons.pt,
            muons.offsets,
            inds,
            sel_ev,
            sel_mu)
        z[:] = target[:]
        
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
        jet = dataset.structs["Jet"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        sel_jet = (jet.pt > 10)
        muons_matched_to_jet = self.ha.mask_deltar_first(
            muons, sel_mu, jet,
            sel_jet, 0.3
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
        self.ha.histogram_from_vector(muons.pt, weights, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32))
        return muons.numevents()

    def test_kernel_histogram_from_vector_several(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        mask = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        weights = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.float32)
        variables = [
            (muons.pt, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32)),
            (muons.eta, self.NUMPY_LIB.linspace(-4,4,100, dtype=self.NUMPY_LIB.float32)),
            (muons.phi, self.NUMPY_LIB.linspace(-4,4,100, dtype=self.NUMPY_LIB.float32)),
            (muons.mass, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32)),
            (muons.charge, self.NUMPY_LIB.linspace(-1,1,3, dtype=self.NUMPY_LIB.float32)),
        ]
        ret = self.ha.histogram_from_vector_several(variables, weights, mask)
        return muons.numevents()

    def test_timing(self):
        with open("kernel_benchmarks.txt", "a") as of:
            for i in range(5):
                ret = self.run_timing()
                of.write(json.dumps(ret) + '\n')

    def run_timing(self):
        ds = self.dataset

        print("Testing memory transfer speed")
        t0 = time.time()
        for i in range(5):
            ds.move_to_device(self.NUMPY_LIB)
        t1 = time.time()
        dt = (t1 - t0)/5.0

        ret = {
            "use_cuda": self.use_cuda, "num_threads": numba.config.NUMBA_NUM_THREADS,
            "use_avx": numba.config.ENABLE_AVX, "num_events": ds.numevents(),
            "memsize": ds.memsize()
        }

        print("Memory transfer speed: {0:.2f} MHz, event size {1:.2f} bytes, data transfer speed {2:.2f} MB/s".format(
            ds.numevents() / dt / 1000.0 / 1000.0, ds.eventsize(), ds.memsize()/dt/1000/1000))
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

        t = self.time_kernel(self.test_kernel_get_in_offsets)
        print("get_in_offsets {0:.2f} MHz".format(t/1000/1000))
        ret["get_in_offsets"] = t/1000/1000

        t = self.time_kernel(self.test_kernel_mask_deltar_first)
        print("mask_deltar_first {0:.2f} MHz".format(t/1000/1000))
        ret["mask_deltar_first"] = t/1000/1000
        
        t = self.time_kernel(self.test_kernel_select_muons_opposite_sign)
        print("select_muons_opposite_sign {0:.2f} MHz".format(t/1000/1000))
        ret["select_muons_opposite_sign"] = t/1000/1000
        
        t = self.time_kernel(self.test_kernel_histogram_from_vector)
        print("histogram_from_vector {0:.2f} MHz".format(t/1000/1000))
        ret["histogram_from_vector"] = t/1000/1000
        
        t = self.time_kernel(self.test_kernel_histogram_from_vector_several)
        print("histogram_from_vector_several {0:.2f} MHz".format(t/1000/1000))
        ret["histogram_from_vector_several"] = t/1000/1000
        
        return ret 

dataset = TestKernels.load_dataset(TestKernels.NUMPY_LIB)
if __name__ == "__main__":
    unittest.main()
