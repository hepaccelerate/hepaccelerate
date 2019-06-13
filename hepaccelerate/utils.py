import sys
import os
import time
import json
import numpy as np
from collections import OrderedDict
import json

import uproot

import numba
from numba import types
from numba.typed import Dict

import awkward

"""
Choose either the CPU(numpy) or GPU/CUDA(cupy) backend.

Args:
    use_cuda: True if you want to use CUDA, False otherwise
    verbose: Print messages

Returns: handle to numpy or cupy library, handle to hepaccelerate backend module
"""
def choose_backend(use_cuda=False, verbose=False):
    if use_cuda:
        if verbose:
            print("Using the GPU CUDA backend")
        import cupy
        NUMPY_LIB = cupy
        import hepaccelerate.backend_cuda as ha
        NUMPY_LIB.searchsorted = ha.searchsorted
    else:
        if verbose:
            print("Using the numpy CPU backend")
        import numpy as numpy
        NUMPY_LIB = numpy
        import hepaccelerate.backend_cpu as ha
        NUMPY_LIB.asnumpy = numpy.array
    return NUMPY_LIB, ha

"""
Simple one-dimensional histogram from a content array (weighted),
content array with squared weights and an edge array.
"""
class Histogram:
    def __init__(self, contents, contents_w2, edges):
        self.contents = np.array(contents)
        self.contents_w2 = np.array(contents_w2)
        self.edges = np.array(edges)
    
    def __add__(self, other):
        assert(np.all(self.edges == other.edges))
        return Histogram(self.contents +  other.contents, self.contents_w2 +  other.contents_w2, self.edges)

    def __mul__(self, number):
        return Histogram(number * self.contents, number * number * self.contents_w2, self.edges)

    def __rmul__(self, number):
        return self.__mul__(number)

    @staticmethod
    def from_dict(d):
        return Histogram(d["contents"], d["contents_w2"], d["edges"])

"""
Collects multiple jagged arrays together into a logical "struct".
"""
class JaggedStruct(object):
    def __init__(self, offsets, attrs_data, prefix, numpy_lib, attr_names_dtypes):
        self.numpy_lib = numpy_lib
        self.hepaccelerate_backend = None
        
        self.offsets = offsets
        self.attrs_data = attrs_data
        self.attr_names_dtypes = attr_names_dtypes
        self.prefix = prefix
        
        num_items = None
        for (k, v) in self.attrs_data.items():
            num_items_next = len(v)
            if num_items and num_items != num_items_next:
                raise AttributeError("Mismatched attribute {0}".format(k))
            else:
                num_items = num_items_next
        self.num_items = num_items

        raise_fmterror = False
        bad_branches = []
        for branch, dtype in self.attr_names_dtypes:
            arr = self.attrs_data[branch.replace(self.prefix, "")]
            if arr.dtype != getattr(self.numpy_lib, dtype):
                raise_fmterror = True
                bad_branches += [branch]
                print("ERROR reading the ROOT TTree: branch {0} declared as {1} but was {2}".format(branch, dtype, arr.dtype), file=sys.stderr)
        if raise_fmterror:
            raise Exception("Declared data structure did not match ROOT file. The Dataset data structure did not match the ROOT TTree for the branches: {0}, please see above for more details.".format(bad_branches))
    
        self.masks = {}
        self.masks["all"] = self.make_mask()
    
    def make_mask(self):
        return self.numpy_lib.ones(self.num_items, dtype=self.numpy_lib.bool)
    
    def mask(self, name):
        if not name in self.masks.keys():
            self.masks[name] = self.make_mask()
        return self.masks[name]
    
    def memsize(self):
        size_tot = self.offsets.size
        for k, v in self.attrs_data.items():
            size_tot += v.nbytes
        return size_tot
    
    def numevents(self):
        return len(self.offsets) - 1

    def numobjects(self):
        for k, v in self.attrs_data.items():
            return len(self.attrs_data[k])
    
    @staticmethod
    def from_arraydict(arraydict, prefix, numpy_lib, attr_names_dtypes):
        ks = [k for k in arraydict.keys() if prefix in str(k, 'ascii')]
        k0 = ks[0]
        return JaggedStruct(
            numpy_lib.array(arraydict[k0].offsets),
            {str(k, 'ascii').replace(prefix, ""): numpy_lib.array(v.content)
             for (k,v) in arraydict.items()},
            prefix, numpy_lib, attr_names_dtypes
        )

    def save(self, path):
        for attr, dtype in self.attr_names_dtypes + [("offsets", "uint64")]:
            attr = attr.replace(self.prefix, "")
            arr = getattr(self, attr)
            m = np.memmap(path + ".{0}.mmap".format(attr), dtype=dtype, mode='write',
                shape=(len(arr))
            )
            m[:] = arr[:]
    
    @staticmethod 
    def load(path, prefix, attr_names_dtypes, numpy_lib):
        attrs_data = {}
        offsets = None
        for attr, dtype in attr_names_dtypes + [("offsets", "uint64")]:
            attr = attr.replace(prefix, "")
            m = np.memmap(path + ".{0}.mmap".format(attr), dtype=dtype, mode='r')
            arr = numpy_lib.array(m)
            if attr == "offsets":
                offsets = arr
            else:
                attrs_data[attr] = arr
        return JaggedStruct(offsets, attrs_data, prefix, numpy_lib, attr_names_dtypes)

    def move_to_device(self, numpy_lib):
        self.numpy_lib = numpy_lib
        new_offsets = self.numpy_lib.array(self.offsets)
        new_attrs_data = {k: self.numpy_lib.array(v) for k, v in self.attrs_data.items()}
        self.offsets = new_offsets
        self.attrs_data = new_attrs_data
 
    def __getattr__(self, attr):
        if attr in self.attrs_data.keys():
            return self.attrs_data[attr]
        return self.__getattribute__(attr)
    
    def __repr__(self):
        s = "JaggedStruct({0}, Nev={1}, Nobj={2}, attrs=[{3}])".format(
            self.prefix, self.numevents(), self.numobjects(), ", ".join(sorted(self.attrs_data.keys())))
        return s

    def compact_struct(self, event_mask):
        assert(len(event_mask) == self.numevents())
        
        new_attrs_data = {}
        new_offsets = None 
        for attr_name, flat_array in self.attrs_data.items():

            #https://github.com/scikit-hep/awkward-array/issues/130
            offsets_int64 = self.offsets.view(np.int64)
            if np.any(offsets_int64 != self.offsets):
                raise Exception("Failed to convert offsets from uint64 to int64")

            ja = awkward.JaggedArray.fromoffsets(offsets_int64, flat_array)

            ja_reduced = ja[event_mask].compact()
            new_attrs_data[attr_name] = ja_reduced.content
            new_offsets = ja_reduced.offsets
        return JaggedStruct(new_offsets, new_attrs_data, self.prefix, self.numpy_lib, self.attr_names_dtypes)

    def select_nth(self, idx, event_mask=None, object_mask=None, attributes=None):
        if type(idx) == int:
            inds = self.numpy_lib.zeros(self.numevents(), dtype=self.numpy_lib.int32)
            inds[:] = idx
        elif type(idx) == self.numpy_lib.ndarray:
            inds = idx
        else:
            raise TypeError("idx must be int or numpy/cupy ndarray")

        if event_mask is None:
            event_mask = self.numpy_lib.ones(self.numevents(), dtype=self.numpy_lib.bool)

        if object_mask is None:
            object_mask = self.numpy_lib.ones(self.numobjects(), dtype=self.numpy_lib.bool)

        if attributes is None:
            attributes = self.attrs_data.keys()

        new_attrs = {
            attr_name: self.hepaccelerate_backend.get_in_offsets(getattr(self, attr_name), self.offsets, inds, event_mask, object_mask)
            for attr_name in attributes
        }

        return new_attrs

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Histogram):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

"""
Dictionary that can be added to others using +
"""
class Results(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __add__(self, other):
        d0 = self
        d1 = other
        
        d_ret = Results({})
        k0 = set(d0.keys())
        k1 = set(d1.keys())

        for k in k0.intersection(k1):
            d_ret[k] = d0[k] + d1[k]

        for k in k0.difference(k1):
            d_ret[k] = d0[k]

        for k in k1.difference(k0):
            d_ret[k] = d1[k]

        return d_ret
    
    def save_json(self, outfn):
        with open(outfn, "w") as fi:
            fi.write(json.dumps(dict(self), indent=2, cls=NumpyEncoder))

"""
Generic uproot dataset
"""
class BaseDataset(object):
    def __init__(self, filenames, arrays_to_load, treename):
        self.filenames = filenames
        self.arrays_to_load = arrays_to_load
        self.data_host = []
        self.treename = treename

    def preload(self, nthreads=1, verbose=False):
        t0 = time.time()
        for ifn, fn in enumerate(self.filenames):
            fi = uproot.open(fn)
            tt = fi.get(self.treename)
            if nthreads > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=nthreads) as executor:
                    arrs = tt.arrays(self.arrays_to_load, executor=executor)
            else:
                arrs = tt.arrays(self.arrays_to_load)
            self.data_host += [arrs]
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("Loaded {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz".format(len(self), dt, len(self)/dt))

    def num_events_raw(self):
        nev = 0
        for arrs in self.data_host:
            k0 = list(arrs.keys())[0]
            nev += len(arrs[k0])
        return nev

    def __len__(self):
        return self.num_events_raw()

"""
Dataset that supports caching
"""
class Dataset(BaseDataset):
    numpy_lib = np
    
    def __init__(self, name, filenames, datastructures,
        datapath="", cache_location="", treename="Events"):
        
        self.datapath = datapath
        self.name = name

        arrays_to_load = []
        for ds_item, ds_vals in datastructures.items():
            for branch, dtype in ds_vals:
                arrays_to_load += [branch]
        super(Dataset, self).__init__(filenames, arrays_to_load, treename)
        
        self.eventvars_dtypes = datastructures.get("EventVariables")
        self.names_eventvars = [evvar for evvar, dtype in self.eventvars_dtypes] 
        self.structs_dtypes = {k: v for (k, v) in datastructures.items() if k != "EventVariables"}
        self.names_structs = sorted(self.structs_dtypes.keys())
        self.cache_location = cache_location
         
        #lists of data, one per file
        self.structs = {}
        for structname in self.names_structs:
            self.structs[structname] = []
        self.eventvars = []

        self.func_filename_precompute = None
        self.cache_metadata = []

    def move_to_device(self, numpy_lib):
        for ifile in range(len(self.filenames)):
            for structname in self.names_structs:
                self.structs[structname][ifile].move_to_device(numpy_lib)
            for evvar in self.names_eventvars:
                self.eventvars[ifile][evvar] = numpy_lib.array(self.eventvars[ifile][evvar])

    def memsize(self):
        tot = 0
        for ifile in range(len(self.filenames)):
            for structname in self.names_structs:
                tot += self.structs[structname][ifile].memsize()
            for evvar in self.names_eventvars:
                tot += self.eventvars[ifile][evvar].nbytes
        return tot

    def eventsize(self):
        nev = self.numevents()
        mem = self.memsize()
        return mem/nev
 
    def __repr__(self):
        s = "DDataset(files={0}, events={1}, {2})".format(len(self.filenames), len(self), ", ".join(self.structs.keys()))
        return s

    def get_cache_dir(self, fn):
        return self.cache_location + fn.replace(self.datapath, "")

    def printout(self):
        s = str(self) 
        for structname in self.structs.keys():
            s += "\n"
            s += "  {0}({1}, {2})".format(structname, self.num_objects_loaded(structname), ", ".join(self.structs[structname][0].attrs_data.keys()))
        s += "\n"
        s += "  EventVariables({0}, {1})".format(len(self), ", ".join(self.names_eventvars))
        return s    

    def load_root(self, nthreads=1, verbose=False):
        self.preload()
        self.make_objects()
        self.cache_metadata = self.create_cache_metadata()

    def preload(self, nthreads=1, verbose=False):
        super(Dataset, self).preload(nthreads, verbose)
 
    def build_structs(self, prefix): 
        struct_array = [
            JaggedStruct.from_arraydict(
                {k: v for k, v in arrs.items() if prefix + "_" in str(k)},
                prefix + "_", self.numpy_lib, self.structs_dtypes[prefix]
            ) for arrs in self.data_host
        ]
        return struct_array

    def make_objects(self, verbose=False):
        t0 = time.time()

        for structname in self.names_structs:
            self.structs[structname] = self.build_structs(structname)

        self.eventvars = [{
            k: self.numpy_lib.array(data[bytes(k, encoding='ascii')])
                for k in self.names_eventvars
        } for data in self.data_host]
  
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("Made objects in {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz".format(len(self), dt, len(self)/dt))

    def analyze(self, analyze_data, verbose=False, **kwargs):
        t0 = time.time()
        rets = []
        for ifile in range(len(self.filenames)):
            data = {}
            for structname in self.names_structs:
                data[structname] = self.structs[structname][ifile]
            data["num_events"] = self.structs[structname][ifile].numevents()
            data["eventvars"] = self.eventvars[ifile]
            ret = analyze_data(data, **kwargs)
            rets += [ret]
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("analyze: processed analysis with {0:.2E} events in {1:.1f} seconds, {2:.2E} Hz, {3:.2E} MB/s".format(len(self), dt, len(self)/dt, self.memsize()/dt/1024.0/1024.0))
        return sum(rets, Results({}))

    def to_cache(self, nthreads=1, verbose=False):
        t0 = time.time()
        
        if nthreads == 1:
            for ifn in range(len(self.filenames)):
                self.to_cache_worker(ifn)
        else:
            from concurrent.futures import ThreadPoolExecutor
            results = []
            with ThreadPoolExecutor(max_workers=nthreads) as executor:
                executor.map(self.to_cache_worker, range(len(self.filenames)))
        
         
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("to_cache: created cache for {1:.2E} events in {0:.1f} seconds, speed {2:.2E} Hz".format(
                dt, len(self), len(self)/dt
            ))

    def filename_to_cachedir(self, filename):
        bfn = os.path.basename(filename).replace(".root", "")
        dn = os.path.dirname(self.get_cache_dir(filename))
        return bfn, dn

    @staticmethod
    def makedir_safe(dn):
        #maybe directory was already created by another worker
        try:
            os.makedirs(dn)
        except FileExistsError as e:
            pass

    def create_cache_metadata(self):
        
        ret_cache_metadata = []
        for ifn in range(len(self.filenames)):
            fn = self.filenames[ifn] 
            
            precomputed_results = {}
            if not (self.func_filename_precompute is None):
                precomputed_results = self.func_filename_precompute(fn)
            cache_metadata = {"filename": fn, "precomputed_results": precomputed_results, "numevents": self.numevents()}
            ret_cache_metadata += [cache_metadata]
        return ret_cache_metadata

 
    def to_cache_worker(self, ifn):
        fn = self.filenames[ifn] 
        bfn, dn = self.filename_to_cachedir(fn)
        self.makedir_safe(dn)

        for structname in self.names_structs:
            self.structs[structname][ifn].save(os.path.join(dn, bfn + ".{0}".format(structname)))

        for attr, dtype in self.eventvars_dtypes:
            arr = self.eventvars[ifn][attr]
            m = np.memmap(os.path.join(dn, bfn + ".{0}.mmap".format(attr)),
                dtype=dtype, mode='write', shape=(len(arr))
            )
            m[:] = arr[:]
        
        cache_metadata = self.cache_metadata[ifn] 
        with open(os.path.join(dn, bfn + ".cache.json".format(attr)), "w") as fi:
            fi.write(json.dumps(cache_metadata, indent=2))

        return

    def from_cache(self, verbose=False, executor=None):
        t0 = time.time()

        if not executor:
            for ifn in range(len(self.filenames)):
                ifn, loaded_structs, eventvars, cache_metadata = self.from_cache_worker(ifn)
                for structname in self.names_structs:
                    self.structs[structname] += [loaded_structs[structname]]
                self.eventvars += [eventvars]
                self.cache_metadata = [cache_metadata]
        else:
            results = executor.map(self.from_cache_worker, range(len(self.filenames)))
            results = list(sorted(results, key=lambda x: x[0]))
            for structname in self.names_structs:
                self.structs[structname] = [r[1][structname] for r in results]
            self.eventvars = [r[2] for r in results] 
            self.cache_metadata = [r[3] for r in results]
        
        t1 = time.time()
        dt = t1 - t0
        if verbose:
            print("from_cache: loaded cache for {1:.2E} events in {0:.1f} seconds, speed {2:.2E} Hz, {3:.2E} MB/s".format(
                dt, len(self), len(self)/dt, self.memsize()/dt/1024.0/1024.0
            ))

    def from_cache_worker(self, ifn):
        fn = self.filenames[ifn]
        bfn = os.path.basename(fn).replace(".root", "")
        
        dn = os.path.dirname(self.get_cache_dir(fn))

        loaded_structs = {}
        for struct in self.names_structs:
            loaded_structs[struct]= JaggedStruct.load(os.path.join(dn, bfn+".{0}".format(struct)), struct+"_", self.structs_dtypes[struct], self.numpy_lib)
        
        eventvars = {}
        for attr, dtype in self.eventvars_dtypes:
            m = np.memmap(os.path.join(dn, bfn + ".{0}.mmap".format(attr)),
                dtype=dtype, mode='r'
            )
            eventvars[attr] = self.numpy_lib.array(m)
        
        with open(os.path.join(dn, bfn + ".cache.json".format(attr)), "r") as fi:
            cache_metadata = json.load(fi)

        return ifn, loaded_structs, eventvars, cache_metadata
 
    def num_objects_loaded(self, structname):
        n_objects = 0
        for ifn in range(len(self.filenames)):
            n_objects += self.structs[structname][ifn].numobjects()
        return n_objects
   
    def numevents(self):
        structname = list(self.structs.keys())[0]
        return self.num_events_loaded(structname)

    def num_events_loaded(self, structname):
        n_events = 0
        for ifn in range(len(self.filenames)):
            n_events += self.structs[structname][ifn].numevents()
        return n_events

    def map(self, func):
        rets = []
        for ifile in range(len(self.filenames)):
            ret = func(self, ifile)
            rets += [ret]
        return rets

    def compact(self, masks):
        for ifile in range(len(self.filenames)):
            for structname in self.names_structs:
                self.structs[structname][ifile] = self.structs[structname][ifile].compact_struct(masks[ifile])
            for evvar in self.names_eventvars:
                self.eventvars[ifile][evvar] = self.eventvars[ifile][evvar][masks[ifile]]

    def __len__(self):
        n_events_raw = self.num_events_raw()
        n_events_loaded = {k: self.num_events_loaded(k) for k in self.names_structs}
        kfirst = self.names_structs[0]
        if n_events_raw == 0:
            return n_events_loaded[kfirst]
        else:
            return n_events_raw

###
### back-ported from https://github.com/CoffeaTeam/coffea
### The code below follows the coffea license.

# BSD 3-Clause License

# Copyright (c) 2018, Fermilab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class LumiMask(object):
    """
        Class that parses a 'golden json' into an efficient valid lumiSection lookup table
        Instantiate with the json file, and call with an array of runs and lumiSections, to
        return a boolean array, where valid lumiSections are marked True
    """
    def __init__(self, jsonfile, numpy_lib, backend):
        with open(jsonfile) as fin:
            goldenjson = json.load(fin)
        self._masks = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )

        self.backend = backend
        self.numpy_lib = numpy_lib

        for run, lumilist in goldenjson.items():
            run = int(run)
            mask = self.numpy_lib.array(lumilist).flatten()
            mask[::2] -= 1
            self._masks[int(run)] = mask

    def __call__(self, runs, lumis):
        mask_out = self.numpy_lib.zeros(dtype='bool', shape=runs.shape)
        LumiMask.apply_run_lumi_mask(self._masks, runs, lumis, mask_out, self.backend)
        return mask_out

    @staticmethod
    def apply_run_lumi_mask(masks, runs, lumis, mask_out, backend):
        backend.apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out)

class LumiData(object):
    """
        Class to hold and parse per-lumiSection integrated lumi values
        as returned by brilcalc, e.g. with a command such as:
        $ brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
                -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
                -u /pb --byls --output-style csv -i Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt > lumi2017.csv
    """
    def __init__(self, lumi_csv):
        self._lumidata = np.loadtxt(lumi_csv, delimiter=',', usecols=(0,1,6,7), converters={
            0: lambda s: s.split(b':')[0],
            1: lambda s: s.split(b':')[0], # not sure what lumi:0 means, appears to be always zero (DAQ off before beam dump?)
        })
        self.index = Dict.empty(
            key_type = types.Tuple([types.uint32, types.uint32]),
            value_type = types.float64
        )
        self.build_lumi_table()
    
    def build_lumi_table(self):
        runs = self._lumidata[:, 0].astype('u4')
        lumis = self._lumidata[:, 1].astype('u4')
        LumiData.build_lumi_table_kernel(runs, lumis, self._lumidata, self.index)

    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def build_lumi_table_kernel(runs, lumis, lumidata, index):
        for i in range(len(runs)):
            run = runs[i]
            lumi = lumis[i]
            index[(run, lumi)] = float(lumidata[i, 2])
            
    def get_lumi(self, runslumis):
        """
            Return integrated lumi
            runlumis: 2d numpy array of [[run,lumi], [run,lumi], ...] or LumiList object
        """
        tot_lumi = np.zeros((1, ), dtype=np.float64)
        LumiData.get_lumi_kernel(runslumis[:, 0], runslumis[:, 1], self.index, tot_lumi)
        return tot_lumi[0]
    
    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def get_lumi_kernel(runs, lumis, index, tot_lumi):
        ks_done = set()
        for iev in range(len(runs)):
            run = runs[iev]
            lumi = lumis[iev]
            k = (run, lumi)
            if not k in ks_done:
                ks_done.add(k)
                tot_lumi[0] += index.get(k, 0)
