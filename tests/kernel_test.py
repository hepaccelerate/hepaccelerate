import time
import os
import requests
import unittest
import numpy as np
import json
import sys
import numba
import uproot

import hepaccelerate
import hepaccelerate.kernels as kernels
from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend

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
            if iblock % 10000 == 0:
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

def load_dataset(numpy_lib, num_iter=1):
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
    dataset = Dataset(
        "nanoaod", num_iter*["./data/nanoaod_test.root"],
        datastructures, treename="Events", datapath="")
  
    dataset.load_root(verbose=True)
    dataset.merge_inplace(verbose=True)
    print("dataset has {0} events, {1:.2f} MB".format(dataset.numevents(), dataset.memsize()/1000/1000))
    dataset.move_to_device(numpy_lib, verbose=True)
    return dataset

@numba.njit
def verify_set_in_offsets(offsets_np, inds_np, arr_np, target_np):
    for iev in range(len(offsets_np) - 1):
        nmu = 0
        for imu in range(offsets_np[iev], offsets_np[iev+1]):
            if nmu == inds_np[iev]:
                if arr_np[imu] != target_np[imu]:
                    print("Mismatch detected in iev,imu", iev, imu)
                    return False
            nmu += 1
    return True 

@numba.njit
def verify_get_in_offsets(offsets_np, inds_np, arr_np, target_np, z_np):
    for iev in range(len(offsets_np) - 1):
        nmu = 0
        
        #Event that had no muons
        if offsets_np[iev] == offsets_np[iev+1]:
            if z_np[iev] !=  0:
                print("Mismatch detected", iev)
                return False

        for imu in range(offsets_np[iev], offsets_np[iev+1]):
            if nmu == inds_np[iev]:
                a = target_np[iev] != z_np[iev]
                b = z_np[iev] != arr_np[imu]
                if a or b:
                    print("Mismatch detected", iev, imu)
                    return False
            nmu += 1
    return True

@numba.njit
def verify_broadcast(offsets_np, vals_ev_np, vals_obj_np):
    for iev in range(len(offsets_np) - 1):
        for iobj in range(offsets_np[iev], offsets_np[iev+1]):
            if vals_obj_np[iobj] != vals_ev_np[iev]:
                print("Mismatch detected in ", iev, iobj)
                return False
    return True 

class TestKernels(unittest.TestCase):
    @classmethod
    def setUpClass(self):    
        self.NUMPY_LIB, self.ha = choose_backend(use_cuda=USE_CUDA)
        self.use_cuda = USE_CUDA
        self.dataset = load_dataset(self.NUMPY_LIB)

    #def time_kernel(self, test_kernel):
    #    test_kernel()
    #
    #    t0 = time.time()
    #    for i in range(5):
    #        n = test_kernel()
    #    t1 = time.time()
    #
    #    dt = (t1 - t0) / 5.0
    #    speed = float(n)/dt
    #    return speed

    def test_kernel_sum_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        z = kernels.sum_in_offsets(
            self.ha,
            muons.offsets,
            muons.pt,
            sel_ev,
            sel_mu, dtype=self.NUMPY_LIB.float32)
        return muons.numevents()

    def test_kernel_max_in_offsets(self):
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        z = kernels.max_in_offsets(
            self.ha,
            muons.offsets,
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
        z = kernels.get_in_offsets(
            self.ha,
            muons.offsets,
            muons.pt,
            inds,
            sel_ev,
            sel_mu)
        return muons.numevents()

    def test_kernel_set_get_in_offsets(self):
        print("kernel_set_get_in_offsets")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        arr = muons.pt.copy()
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        inds = self.NUMPY_LIB.zeros(muons.numevents(), dtype=self.NUMPY_LIB.int8)
        
        #set the pt of the first muon in each event to 1 
        inds[:] = 0
        target = self.NUMPY_LIB.ones(muons.numevents(), dtype=muons.pt.dtype)

        kernels.set_in_offsets(
            self.ha,
            muons.offsets,
            arr,
            inds,
            target,
            sel_ev,
            sel_mu)

        print("checking set_in_offsets")
        asnp = self.NUMPY_LIB.asnumpy
        self.assertTrue(verify_set_in_offsets(
            asnp(muons.offsets),
            asnp(inds),
            asnp(arr),
            asnp(target)
        ))

        print("checking get_in_offsets")
        z = kernels.get_in_offsets(
            self.ha,
            muons.offsets,
            arr,
            inds,
            sel_ev,
            sel_mu)

        self.assertTrue(verify_get_in_offsets(
            asnp(muons.offsets),
            asnp(inds),
            asnp(arr),
            asnp(target), 
            asnp(z)
        ))
 
        return muons.numevents()

    def test_kernel_simple_cut(self):
        print("kernel_simple_cut")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_mu = muons.pt > 30.0
        return muons.numevents()
    
    def test_kernel_broadcast(self):
        print("kernel_broadcast")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        met_pt = dataset.eventvars[0]["MET_pt"]
        met_pt_permuon = self.NUMPY_LIB.zeros(muons.numobjects(), dtype=self.NUMPY_LIB.float32)
        kernels.broadcast(self.ha, muons.offsets, met_pt, met_pt_permuon)
        self.assertTrue(verify_broadcast(
            self.NUMPY_LIB.asnumpy(muons.offsets),
            self.NUMPY_LIB.asnumpy(met_pt),
            self.NUMPY_LIB.asnumpy(met_pt_permuon),
        ))

        return muons.numevents()

    def test_kernel_mask_deltar_first(self):
        print("kernel_mask_deltar_first")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        jet = dataset.structs["Jet"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        sel_jet = (jet.pt > 10)
        muons_matched_to_jet = kernels.mask_deltar_first(
            self.ha,
            {"offsets": muons.offsets, "eta": muons.eta, "phi": muons.phi},
            sel_mu,
            {"offsets": jet.offsets, "eta": jet.eta, "phi": jet.phi},
            sel_jet, 0.3
        )
        self.assertEqual(len(muons_matched_to_jet), muons.numobjects())
        self.assertEqual(muons_matched_to_jet.sum(), 1698542)
        return muons.numevents()
        
    def test_kernel_select_opposite_sign(self):
        print("kernel_select_opposite_sign")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        sel_ev = self.NUMPY_LIB.ones(muons.numevents(), dtype=self.NUMPY_LIB.bool)
        sel_mu = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        muons_passing_os = kernels.select_opposite_sign(
            self.ha,
            muons.offsets, muons.charge, sel_mu)
        return muons.numevents()
    
    def test_kernel_histogram_from_vector(self):
        print("kernel_histogram_from_vector")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        weights = 2*self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.float32)
        ret = kernels.histogram_from_vector(self.ha, muons.pt, weights, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32))
        self.assertEqual(ret[0][20], 112024.0)
        self.assertEqual(ret[1][20], 2*112024.0)
        self.assertEqual(ret[0][0], 0)
        self.assertEqual(ret[0][-1], 7856.0)

        self.assertEqual(ret[0].sum(), 3894188.0)
        self.assertEqual(ret[1].sum(), 2*3894188.0)
        return muons.numevents()
    
    def test_kernel_histogram_from_vector_masked(self):
        print("kernel_histogram_from_vector_masked")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        weights = 2*self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.float32)
        mask = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        mask[:100] = False
        ret = kernels.histogram_from_vector(self.ha, muons.pt, weights, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32), mask=mask)
        self.assertEqual(ret[0][20], 112014.0)
        self.assertEqual(ret[1][20], 2*112014.0)
        self.assertEqual(ret[0][0], 0)
        self.assertEqual(ret[0][-1], 7856.0)

        self.assertEqual(ret[0].sum(), 3893988.0)
        self.assertEqual(ret[1].sum(), 2*3893988.0)
        return muons.numevents()

    def test_kernel_histogram_from_vector_several(self):
        print("kernel_histogram_from_vector_several")
        dataset = self.dataset
        muons = dataset.structs["Muon"][0]
        mask = self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.bool)
        mask[:100] = False
        weights = 2*self.NUMPY_LIB.ones(muons.numobjects(), dtype=self.NUMPY_LIB.float32)
        variables = [
            (muons.pt, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32)),
            (muons.eta, self.NUMPY_LIB.linspace(-4,4,100, dtype=self.NUMPY_LIB.float32)),
            (muons.phi, self.NUMPY_LIB.linspace(-4,4,100, dtype=self.NUMPY_LIB.float32)),
            (muons.mass, self.NUMPY_LIB.linspace(0,200,100, dtype=self.NUMPY_LIB.float32)),
            (muons.charge, self.NUMPY_LIB.array([-1, 0, 1, 2], dtype=self.NUMPY_LIB.float32)),
        ]
        ret = kernels.histogram_from_vector_several(self.ha, variables, weights, mask)
       
        #number of individual histograms
        self.assertEqual(len(ret), len(variables))
       
        #weights, weights2, bins 
        self.assertEqual(len(ret[0]), 3)
        
        #bin edges
        for ivar in range(len(variables)):
            self.assertEqual(len(ret[ivar][0]), len(variables[ivar][1]) - 1)
       
        #bin contents
        for ivar in range(len(variables)):
            ret2 = kernels.histogram_from_vector(self.ha, variables[ivar][0], weights, variables[ivar][1], mask=mask)
            for ibin in range(len(ret[ivar][0])):
                self.assertEqual(ret[ivar][0][ibin], ret2[0][ibin])
                self.assertEqual(ret[ivar][1][ibin], ret2[1][ibin])

        return muons.numevents()

    def test_coordinate_transformation(self):
        print("coordinate_transformation")
        #Don't test the scalar ops on GPU
        if not USE_CUDA:
            px, py, pz, e = kernels.spherical_to_cartesian(self.ha, 100.0, 0.2, 0.4, 100.0)
            pt, eta, phi, mass = kernels.cartesian_to_spherical(self.ha, px, py, pz, e)
            self.assertAlmostEqual(pt, 100.0, 2)
            self.assertAlmostEqual(eta, 0.2, 2)
            self.assertAlmostEqual(phi, 0.4, 2)
            self.assertAlmostEqual(mass, 100.0, 2)

        pt_orig = self.NUMPY_LIB.array([100.0, 20.0], dtype=self.NUMPY_LIB.float32)
        eta_orig = self.NUMPY_LIB.array([0.2, -0.2], dtype=self.NUMPY_LIB.float32)
        phi_orig = self.NUMPY_LIB.array([0.4, -0.4], dtype=self.NUMPY_LIB.float32)
        mass_orig = self.NUMPY_LIB.array([100.0, 20.0], dtype=self.NUMPY_LIB.float32)

        px, py, pz, e = kernels.spherical_to_cartesian(self.ha, pt_orig, eta_orig, phi_orig, mass_orig)
        pt, eta, phi, mass = kernels.cartesian_to_spherical(self.ha, px, py, pz, e)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(pt[0]), 100.0, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(eta[0]), 0.2, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(phi[0]), 0.4, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(mass[0]), 100, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(pt[1]), 20, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(eta[1]), -0.2, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(phi[1]), -0.4, 2)
        self.assertAlmostEqual(self.NUMPY_LIB.asnumpy(mass[1]), 20, 2)

        #Don't test the scalar ops on GPU
        if not USE_CUDA:
            pt_tot, eta_tot, phi_tot, mass_tot = self.ha.add_spherical(pt_orig, eta_orig, phi_orig, mass_orig)
            self.assertAlmostEqual(pt_tot, 114.83390378237536)
            self.assertAlmostEqual(eta_tot, 0.13980652560764573)
            self.assertAlmostEqual(phi_tot, 0.2747346427265487)
            self.assertAlmostEqual(mass_tot, 126.24366428840153)

if __name__ == "__main__":
    if "--debug" in sys.argv:
        unittest.findTestCases(sys.modules[__name__]).debug()
    else:
        unittest.main()
