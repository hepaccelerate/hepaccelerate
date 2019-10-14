from hepaccelerate.utils import Results, Dataset, Histogram, choose_backend, JaggedStruct
import uproot
import numpy
import numpy as np
import unittest
import os
from uproot_methods.classes.TH1 import from_numpy

USE_CUDA = bool(int(os.environ.get("HEPACCELERATE_CUDA", 0)))

class TestJaggedStruct(unittest.TestCase):
    def test_jaggedstruct(self):
        attr_names_dtypes = [("Muon_pt", "float64")]
        js = JaggedStruct([0,2,3], {"pt": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)}, "Muon_", np, attr_names_dtypes)
        js.attr_names_dtypes = attr_names_dtypes
        js.save("cache")
    
        js2 = JaggedStruct.load("cache", "Muon_", attr_names_dtypes, np)
    
        np.all(js.offsets == js2.offsets)
        for k in js.attrs_data.keys():
            np.all(getattr(js, k) == getattr(js2, k))

class TestHistogram(unittest.TestCase):
    NUMPY_LIB, ha = choose_backend(use_cuda=USE_CUDA)

    def test_histogram(self):
        np = TestHistogram.NUMPY_LIB
        data = np.array([2,3,4,5,6,7], dtype=np.float32)
        data[data<2] = 0
        weights = np.ones_like(data, dtype=np.float32)
        w, w2, e = self.ha.histogram_from_vector(data, weights, np.array([0,1,2,3,4,5], dtype=np.float32))
        npw, npe = np.histogram(data, np.array([0,1,2,3,4,5]))
        hr = from_numpy((w, e))
        f = uproot.recreate("test.root")
        f["hist"]  = hr
        
        data = np.random.normal(size=10000)
        data = np.array(data, dtype=np.float32)
        weights = np.ones_like(data, dtype=np.float32)
        w, w2, e = self.ha.histogram_from_vector(data, weights, np.linspace(-1,1,100, dtype=np.float32))
        hr = from_numpy((w, e))
        f["hist2"]  = hr
        f.close()

    def test_histogram_several(self):
        np = TestHistogram.NUMPY_LIB
        data = np.array([2,3,4,5,6,7], dtype=np.float32)
        mask = data>=2
        data[self.NUMPY_LIB.invert(mask)] = 0
        weights = np.ones_like(data, dtype=np.float32)
        bins = np.array([0,1,2,3,4,5], dtype=np.float32)
        w, w2, e = self.ha.histogram_from_vector(data, weights, bins)

        ws, ws2, all_bins = self.ha.histogram_from_vector_several([(data, bins), (data, bins)], weights, mask)
        assert(numpy.all(w == ws[0]))
        assert(numpy.all(w == ws[1]))
        assert(numpy.all(w2 == ws2[0]))
        assert(numpy.all(w2 == ws2[1]))

class TestDataset(unittest.TestCase):
    NUMPY_LIB, ha = choose_backend(use_cuda=USE_CUDA)
    
    @staticmethod
    def load_dataset(num_iter=1):
        #fi = uproot.open("data/HZZ.root")
        #print(fi.keys())
        #print(fi.get("events").keys())
        
        datastructures = {
                "Muon": [
                    ("Muon_Px", "float32"),
                    ("Muon_Py", "float32"),
                    ("Muon_Pz", "float32"), 
                    ("Muon_E", "float32"),
                    ("Muon_Charge", "int32"),
                    ("Muon_Iso", "float32")
                ],
                "Jet": [
                    ("Jet_Px", "float32"),
                    ("Jet_Py", "float32"),
                    ("Jet_Pz", "float32"),
                    ("Jet_E", "float32"),
                    ("Jet_btag", "float32"),
                    ("Jet_ID", "bool")
                ],
                "EventVariables": [
                    ("NPrimaryVertices", "int32"),
                    ("triggerIsoMu24", "bool"),
                    ("EventWeight", "float32")
                ]
            }
        dataset = Dataset("HZZ", num_iter*["data/HZZ.root"], datastructures, cache_location="./mycache/", treename="events", datapath="")
        assert(dataset.filenames[0] == "data/HZZ.root")
        assert(len(dataset.filenames) == num_iter)
        assert(len(dataset.structs["Jet"]) == 0)
        assert(len(dataset.eventvars) == 0)
        return dataset

    def setUp(self):
        self.dataset = self.load_dataset()

    def test_dataset_to_cache(self):
        dataset = self.dataset
    
        dataset.load_root()
        assert(len(dataset.data_host) == 1)
        
        assert(len(dataset.structs["Jet"]) == 1)
        assert(len(dataset.eventvars) == 1)
    
        dataset.to_cache()
        return dataset
    
    def test_dataset_from_cache(self):
        dataset = self.dataset
        dataset.load_root()
        dataset.to_cache()
        del dataset
        dataset = self.load_dataset()
        dataset.from_cache()
        
        dataset2 = self.load_dataset()
        dataset2.load_root()
    
        assert(dataset.num_objects_loaded("Jet") == dataset2.num_objects_loaded("Jet"))
        assert(dataset.num_events_loaded("Jet") == dataset2.num_events_loaded("Jet"))
   
    @staticmethod
    def map_func(dataset, ifile):
        mu = dataset.structs["Muon"][ifile]
        mu_pt = np.sqrt(mu.Px**2 + mu.Py**2)
        mu_pt_pass = mu_pt > 20
        mask_rows = np.ones(mu.numevents(), dtype=np.bool)
        mask_content = np.ones(mu.numobjects(), dtype=np.bool)
        ret = TestDataset.ha.sum_in_offsets(mu, mu_pt_pass, mask_rows, mask_content, dtype=np.int8) 
        return ret
    
    def test_dataset_map(self):
        dataset = self.load_dataset()
        dataset.load_root()
    
        rets = dataset.map(self.map_func)
        assert(len(rets) == 1)
        assert(len(rets[0]) == dataset.structs["Muon"][0].numevents())
        assert(np.sum(rets[0]) > 0)
        return rets
    
    def test_dataset_compact(self):
        dataset = self.dataset
        dataset.load_root()
    
        memsize1 = dataset.memsize()
        rets = dataset.map(self.map_func)

        #compacting uses JaggedArray functionality and can only be done on the numpy/CPU backend
        dataset.move_to_device(np)
        rets = [TestDataset.NUMPY_LIB.asnumpy(r) for r in rets]
        dataset.compact(rets)
        dataset.move_to_device(TestDataset.NUMPY_LIB)

        memsize2 = dataset.memsize()
        assert(memsize1 > memsize2)
        print("compacted memory size ratio:", memsize2/memsize1)

    @staticmethod
    def precompute_results(filename):
        fi = uproot.open(filename)
        arr = fi.get("events").array("EventWeight")
        return {"EventWeight": arr.sum()}

    def test_dataset_merge_inplace(self):
        num_iter = 10

        ds_multi = self.load_dataset(num_iter=num_iter)
        ds_multi.func_filename_precompute = self.precompute_results

        ds_multi.load_root()
        assert(len(ds_multi.structs["Jet"]) == num_iter)
        njet = ds_multi.num_objects_loaded("Jet")
       
        #compute a per-event jet energy sum taking into account the offsets
        jet_sume = TestDataset.NUMPY_LIB.hstack([TestDataset.ha.sum_in_offsets(
            ds_multi.structs["Jet"][i],
            ds_multi.structs["Jet"][i]["E"],
            TestDataset.NUMPY_LIB.ones(ds_multi.structs["Jet"][i].numevents(), dtype=TestDataset.NUMPY_LIB.bool),
            TestDataset.NUMPY_LIB.ones(ds_multi.structs["Jet"][i].numobjects(), dtype=TestDataset.NUMPY_LIB.bool)
        ) for i in range(num_iter)])

        numevents = ds_multi.numevents()
        EventWeight_total = sum([md["precomputed_results"]["EventWeight"] for md in ds_multi.cache_metadata])
        numevents_total = sum([md["numevents"] for md in ds_multi.cache_metadata])

        ds_multi.merge_inplace()
        assert(len(ds_multi.structs["Jet"]) == 1)
        assert(ds_multi.num_objects_loaded("Jet") == njet)
        jet_sume_merged = TestDataset.ha.sum_in_offsets(
            ds_multi.structs["Jet"][0],
            ds_multi.structs["Jet"][0]["E"],
            TestDataset.NUMPY_LIB.ones(ds_multi.structs["Jet"][0].numevents(), dtype=TestDataset.NUMPY_LIB.bool),
            TestDataset.NUMPY_LIB.ones(ds_multi.structs["Jet"][0].numobjects(), dtype=TestDataset.NUMPY_LIB.bool)
        )
        assert(TestDataset.NUMPY_LIB.all(jet_sume_merged == jet_sume))
        assert(ds_multi.numevents() == numevents)
        assert(abs(ds_multi.cache_metadata[0]["precomputed_results"]["EventWeight"] - EventWeight_total) < 0.01)
        assert(abs(ds_multi.cache_metadata[0]["numevents"] - numevents_total) == 0)

if __name__ == "__main__":
    unittest.main()
