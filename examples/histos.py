from numba import cuda
import numpy as np
import hepaccelerate.backend_cuda as ha
import time

def launch_kernel(i, data, weights, bins, nblocks, nthreads):
    stream = cuda.stream()
    ha.fill_histogram[nblocks, nthreads, stream](data, weights, bins, out_w, out_w2)
    return stream, out_w, out_w2

def launch_kernel_2(data, weights, bins):
    out_w, _, _ = ha.histogram_from_vector(data, weights, bins)
    return out_w

     
def run_several(N=1000):
    stream_list = [] 
    results = []

    nblocks = 32
    nthreads = 256
    data_stacked = []
    for i in range(N):
        data_stacked += [data]
    data_stacked = cp.stack(data_stacked, axis=0)
    out_w = cp.zeros((N, nblocks, len(bins) - 1), cp.float32)
    out_w2 = cp.zeros((N, nblocks, len(bins) - 1), cp.float32)
    bins_stacked = cp.stack([bins for i in range(N)], axis=0)
    nbins = cp.zeros(N, dtype=cp.int32)
    nbins[:] = len(bins)

    t0 = time.time()
    ha.fill_histogram_several[nblocks,nthreads](data_stacked, weights, bins_stacked, nbins, out_w, out_w2)
    cuda.synchronize()
    out_w = cp.asnumpy(out_w.sum(axis=1))

    for i in range(N):
        assert(np.all(out_w[0] == out_w[i]))
    ret = [out_w[i] for i in range(N)]
    t1 = time.time()
    print("several", t1 - t0)
    return ret[0]

def run_default(N=1000):
    results = []
    t0 = time.time()

    for i in range(N): 
        out_w = launch_kernel_2(data, weights, bins)
        results += [out_w]

    t1 = time.time()
    for i in range(len(results)):
        assert(np.all(results[i] == results[0]))
    print("default", t1 - t0)
    return results[0]

if __name__ == "__main__":
    if int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1:
        import cupy as cp
        data = np.random.normal(size=1000000)
        data = cp.array(data, dtype=cp.float32)
        weights = cp.ones_like(data)
        bins = cp.array(cp.linspace(-2, 2, 100), dtype=cp.float32)
        
        N = 1000
        res = []
        res += [run_several(N=N)]
        res += [run_default(N=N)]
        
        for i in range(len(res)):
            assert(np.all(res[i] == res[0]))
    else:
        print("Skipping due to HEPACCELERATE_CUDA=0")
