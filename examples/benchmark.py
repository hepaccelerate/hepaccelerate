from hepaccelerate.utils import choose_backend, LumiData, LumiMask
import time
import numpy as np


def test_histo_on_backend(cp, ha, data):
    data = cp.array(data, dtype=cp.float32)
    weights = cp.ones_like(data)
    bins = cp.array(cp.linspace(-2,2,100), dtype=cp.float32)
    cuda.synchronize()

    results = []
    t0 = time.time() 
    for i in range(Nhisto):
        if i%10==0:
            print("iteration", i)
        contents, _, _ = ha.histogram_from_vector(data, weights, bins)
        results += [contents] 
    cuda.synchronize()
    t1 = time.time()

    for ires in range(len(results)):
        assert(np.all(results[ires] == results[0]))

    print("all results are equal, as expected")

    nbytes = Nhisto * data.nbytes
    speed = nbytes / (t1 - t0) / 1000 / 1000 / 1000

    speed_ev = Nhisto * len(data) / (t1 - t0)
    print("Speed {0:.2f} GB/s, {1:.2E} ev/s".format(speed, speed_ev))

    return results[0]

if __name__ == "__main__":
    Nhisto = 100
    
    np.random.seed(0)
    data = np.random.normal(size=10000000)
    
    cp, ha = choose_backend(False)
    print("on CPU")
    res_cpu = test_histo_on_backend(cp, ha, data)
  
    run_gpu = False
    try:
        from numba import cuda
        run_gpu = True
    except Exception as e:
        print(e)

    if run_gpu: 
        cp, ha = choose_backend(True)
        print("on GPU")
        res_gpu = test_histo_on_backend(cp, ha, data)
        assert(np.all(res_gpu == res_cpu))
        print("CPU result is equal to GPU result, as expected")
