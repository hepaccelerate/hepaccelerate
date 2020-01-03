import sys
import warnings
import math
import numpy as np

try:
    from numba import cuda
    import cupy
except ImportError as e:
    print("Could not import cupy or numba.cuda, hepaccelerate.backend_cuda not usable", file=sys.stderr)
    print("Exception: {0}".format(e.msg), file=sys.stderr)

@cuda.jit(device=True)
def spherical_to_cartesian_devfunc(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, e

@cuda.jit(device=True)
def cartesian_to_spherical_devfunc(px, py, pz, e):
    pt = math.sqrt(px**2 + py**2)
    eta = math.asinh(pz / pt)
    phi = math.atan2(py, px)
    mass = math.sqrt(e**2 - px**2 - py**2 - pz**2)
    return pt, eta, phi, mass

@cuda.jit(device=True)
def add_spherical_devfunc(pts, etas, phis, masses):
    px_tot = np.float32(0.0)
    py_tot = np.float32(0.0)
    pz_tot = np.float32(0.0)
    e_tot = np.float32(0.0)
    
    for i in range(len(pts)):
        px, py, pz, e = spherical_to_cartesian(pts[i], etas[i], phis[i], masses[i])
        px_tot += px
        py_tot += py
        pz_tot += pz
        e_tot += e
    return cartesian_to_spherical(px_tot, py_tot, pz_tot, e_tot)

@cuda.jit
def spherical_to_cartesian_kernel(pt, eta, phi, mass, out_px, out_py, out_pz, out_e):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(pt), xstride):
        px, py, pz, e = spherical_to_cartesian_devfunc(pt[i], eta[i], phi[i], mass[i])
        out_px[i] = px
        out_py[i] = py
        out_pz[i] = pz
        out_e[i] = e

@cuda.jit
def cartesian_to_spherical_kernel(px, py, pz, e, out_pt, out_eta, out_phi, out_mass):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(px), xstride):
        pt, eta, phi, mass = cartesian_to_spherical_devfunc(px[i], py[i], pz[i], e[i])
        out_pt[i] = pt
        out_eta[i] = eta
        out_phi[i] = phi
        out_mass[i] = mass

def spherical_to_cartesian(pt, eta, phi, mass):
    out_px = cupy.zeros_like(pt)
    out_py = cupy.zeros_like(pt)
    out_pz = cupy.zeros_like(pt)
    out_e = cupy.zeros_like(pt)
    spherical_to_cartesian_kernel[32, 1024](pt, eta, phi, mass, out_px, out_py, out_pz, out_e)
    cuda.synchronize()
    return out_px, out_py, out_pz, out_e

def cartesian_to_spherical(px, py, pz, e):
    out_pt = cupy.zeros_like(px)
    out_eta = cupy.zeros_like(px)
    out_phi = cupy.zeros_like(px)
    out_mass = cupy.zeros_like(px)
    cartesian_to_spherical_kernel[32, 1024](px, py, pz, e, out_pt, out_eta, out_phi, out_mass)
    cuda.synchronize()
    return out_pt, out_eta, out_phi, out_mass

#This is a naive implementation, need to fix to use mod, but somehow causes OUT_OF_RESOURCE!
@cuda.jit(device=True)
def deltaphi_devfunc(phi1, phi2):
    dphi = phi1 - phi2
    out_dphi = 0 
    if dphi > math.pi:
        dphi = dphi - 2*math.pi
        out_dphi = dphi
    elif (dphi + math.pi) < 0:
        dphi = dphi + 2*math.pi
        out_dphi = dphi
    else:
        out_dphi = dphi
    return out_dphi

#Copied from numba source
@cuda.jit(device=True)
def searchsorted_inner_right(a, v):
    n = len(a)
    lo = np.int32(0)
    hi = np.int32(n)
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] <= (v):
            # mid is too low => go up
            lo = mid + 1
        else:
            # mid is too high, or is a NaN => go down
            hi = mid
    return lo

#Copied from numba source
@cuda.jit(device=True)
def searchsorted_inner_left(a, v):
    n = len(a)
    lo = np.int32(0)
    hi = np.int32(n)
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] < (v):
            # mid is too low => go up
            lo = mid + 1
        else:
            # mid is too high, or is a NaN => go down
            hi = mid
    return lo

@cuda.jit(device=True)
def searchsorted_devfunc_right(bins, val):
    ret = searchsorted_inner_right(bins, val)
    if val < bins[0]:
        ret = 0
    if val >= bins[len(bins)-1]:
        ret = len(bins) - 1
    return ret

@cuda.jit(device=True)
def searchsorted_devfunc_left(bins, val):
    ret = searchsorted_inner_left(bins, val)
    if val < bins[0]:
        return 0
    if val >= bins[len(bins)-1]:
        return len(bins) - 1
    return ret

@cuda.jit
def searchsorted_kernel_right(vals, arr, inds_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    assert(len(vals) == len(inds_out))
    
    for i in range(xi, len(vals), xstride):
        inds_out[i] = searchsorted_devfunc_right(arr, vals[i])

@cuda.jit
def searchsorted_kernel_left(vals, arr, inds_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    assert(len(vals) == len(inds_out))
    
    for i in range(xi, len(vals), xstride):
        inds_out[i] = searchsorted_devfunc_left(arr, vals[i])

@cuda.jit
def fill_histogram_several(data, weights, mask, bins, nbins, nbins_sum, out_w, out_w2):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    bi = cuda.blockIdx.x
    bd = cuda.blockDim.x
    ti = cuda.threadIdx.x
  
    #number of histograms to fill 
    ndatavec = data.shape[0]
 
    for iev in range(xi, data.shape[1], xstride):
        if mask[iev]:
            for ivec in range(ndatavec):
                bin_idx = np.int32(
                    searchsorted_devfunc_right(
                        bins[nbins_sum[ivec]:nbins_sum[ivec+1]],
                        data[ivec, iev]
                    ) - 1
                )
                if bin_idx >= nbins[ivec]:
                    bin_idx = nbins[ivec] - 1
                bin_idx_histo = (ivec, bi, bin_idx)
            
                if bin_idx >=0 and bin_idx < nbins[ivec]:
                    wi = weights[iev]
                    cuda.atomic.add(out_w, bin_idx_histo, wi)
                    cuda.atomic.add(out_w2, bin_idx_histo, wi**2)

@cuda.jit
def fill_histogram(data, weights, bins, out_w, out_w2):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
   
    bi = cuda.blockIdx.x
    bd = cuda.blockDim.x
    ti = cuda.threadIdx.x

    nbins = out_w.shape[1]
    
    for i in range(xi, len(data), xstride):
        bin_idx = searchsorted_devfunc_right(bins, data[i]) - 1
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        bin_idx_histo = (bi, bin_idx)

        if bin_idx >=0 and bin_idx < nbins:
            wi = weights[i]
            cuda.atomic.add(out_w, bin_idx_histo, wi)
            cuda.atomic.add(out_w2, bin_idx_histo, wi**2)

@cuda.jit
def fill_histogram_masked(data, weights, bins, mask, out_w, out_w2):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
   
    bi = cuda.blockIdx.x
    bd = cuda.blockDim.x
    ti = cuda.threadIdx.x

    nbins = out_w.shape[1]
    
    for i in range(xi, len(data), xstride):
        if mask[i]:
            bin_idx = searchsorted_devfunc_right(bins, data[i]) - 1
            if bin_idx >= nbins:
                bin_idx = nbins - 1
            bin_idx_histo = (bi, bin_idx)

            if bin_idx >=0 and bin_idx < nbins:
                wi = weights[i]
                cuda.atomic.add(out_w, bin_idx_histo, wi)
                cuda.atomic.add(out_w2, bin_idx_histo, wi**2)

@cuda.jit
def select_opposite_sign_cudakernel(charges_offsets, charges_content, content_mask_in, content_mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, charges_offsets.shape[0]-1, xstride):
        start = np.uint64(charges_offsets[iev])
        end = np.uint64(charges_offsets[iev + 1])
        
        ch1 = np.int32(0)
        idx1 = np.uint64(0)
        ch2 = np.int32(0)
        idx2 = np.uint64(0)
        
        for imuon in range(start, end):
            if not content_mask_in[imuon]:
                continue
                
            if idx1 == 0 and idx2 == 0:
                ch1 = charges_content[imuon]
                idx1 = imuon
                continue
            else:
                ch2 = charges_content[imuon]
                if (ch2 != ch1):
                    idx2 = imuon
                    content_mask_out[idx1] = 1
                    content_mask_out[idx2] = 1
                    break
    return

@cuda.jit
def sum_in_offsets_cudakernel(offsets, content, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] += content[ielem]

@cuda.jit
def prod_in_offsets_cudakernel(offsets, content, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] *= content[ielem]
            
@cuda.jit
def max_in_offsets_cudakernel(offsets, content, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] > accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum

        
@cuda.jit
def min_in_offsets_cudakernel(offsets, content, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
    
        first = True
        accum = 0
        
        for ielem in range(start, end):
            if mask_content[ielem]:
                if first or content[ielem] < accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum

@cuda.jit
def get_in_offsets_cudakernel(offsets, content, indices, mask_rows, mask_content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
        start = offsets[iev]
        end = offsets[iev + 1]
        
        index_to_get = 0
        for ielem in range(start, end):
            if mask_content[ielem]:
                if index_to_get == indices[iev]:
                    out[iev] = content[ielem]
                    break
                else:
                    index_to_get += 1

@cuda.jit
def set_in_offsets_cudakernel(offsets, content, indices, target, mask_rows, mask_content):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        if not mask_rows[iev]:
            continue
        start = offsets[iev]
        end = offsets[iev + 1]
        
        index_to_set = 0
        for ielem in range(start, end):
            if mask_content[ielem]:
                if index_to_set == indices[iev]:
                    content[ielem] = target[iev]
                    break
                else:
                    index_to_set += 1

@cuda.jit
def broadcast_cudakernel(offsets, content, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, offsets.shape[0]-1, xstride):
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            out[ielem] = content[iev]

@cuda.jit
def get_bin_contents_cudakernel(values, edges, contents, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for i in range(xi, len(values), xstride):
        v = values[i]
        ibin = searchsorted_devfunc_right(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

@cuda.jit
def copyto_dst_indices_cudakernel(dst, src, inds_dst):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for i1 in range(xi, len(src), xstride):
        i2 = inds_dst[i1] 
        dst[i2] = src[i1]

@cuda.jit
def compute_new_offsets_cudakernel(offsets_old, mask_objects, counts, offsets_new):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    for iev in range(xi, len(offsets_old)-1, xstride):
        start = offsets_old[iev]
        end = offsets_old[iev + 1]
        ret = 0
        for ielem in range(start, end):
            if mask_objects[ielem]:
                ret += 1
            counts[iev] = ret

    count_tot = 0
    for iev in range(len(counts)):
        offsets_new[iev] = count_tot
        offsets_new[iev+1] = count_tot + counts[iev]
        count_tot += counts[iev] 

@cuda.jit
def mask_deltar_first_cudakernel(etas1, phis1, mask1, offsets1, etas2, phis2, mask2, offsets2, dr2, mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, len(offsets1)-1, xstride):
        a1 = np.uint64(offsets1[iev])
        b1 = np.uint64(offsets1[iev+1])
        
        a2 = np.uint64(offsets2[iev])
        b2 = np.uint64(offsets2[iev+1])
        
        for idx1 in range(a1, b1):
            if not mask1[idx1]:
                continue
                
            eta1 = np.float32(etas1[idx1])
            phi1 = np.float32(phis1[idx1])
            for idx2 in range(a2, b2):
                if not mask2[idx2]:
                    continue
                eta2 = np.float32(etas2[idx2])
                phi2 = np.float32(phis2[idx2])
                
                deta = abs(eta1 - eta2)
                dphi = deltaphi_devfunc(phi1, phi2)
                
                #if first object is closer than dr2, mask element will be *disabled*
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr

def make_masks(offsets, content, mask_rows, mask_content):
    if mask_rows is None:
        mask_rows = cupy.ones(len(offsets) - 1, dtype=cupy.bool)
    if mask_content is None:
        mask_content = cupy.ones(len(content), dtype=cupy.bool)
    return mask_rows, mask_content

# Kernel wrappers

"""
Find indices to insert vals into arr to preserve order.
"""
def searchsorted(arr, vals, side="right"):
    ret = cupy.zeros_like(vals, dtype=cupy.int32)
    if side == "right":
        searchsorted_kernel_right[32, 1024](vals, arr, ret)
    elif side == "left":
        searchsorted_kernel_left[32, 1024](vals, ret, arr)
    cuda.synchronize()
    return ret 

def broadcast(offsets, content, out):
    broadcast_cudakernel[32,1024](offsets, content, out)
    cuda.synchronize()

def sum_in_offsets(offsets, content, mask_rows=None, mask_content=None, dtype=None):
    if not dtype:
        dtype = content.dtype
    sum_offsets = cupy.zeros(len(offsets) - 1, dtype=dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    sum_in_offsets_cudakernel[32, 1024](offsets, content, mask_rows, mask_content, sum_offsets)
    cuda.synchronize()
    return sum_offsets

def prod_in_offsets(offsets, content, mask_rows=None, mask_content=None, dtype=None):
    if not dtype:
        dtype = content.dtype
    ret = cupy.ones(len(offsets) - 1, dtype=dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    prod_in_offsets_cudakernel[32, 1024](offsets, content, mask_rows, mask_content, ret)
    cuda.synchronize()
    return ret

def max_in_offsets(offsets, content, mask_rows=None, mask_content=None):
    max_offsets = cupy.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    max_in_offsets_cudakernel[32, 1024](offsets, content, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def min_in_offsets(offsets, content, mask_rows=None, mask_content=None):
    max_offsets = cupy.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    min_in_offsets_cudakernel[32, 1024](offsets, content, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def select_opposite_sign(offsets, charges, in_mask):
    out_mask = cupy.zeros_like(charges, dtype=cupy.bool)
    select_opposite_sign_cudakernel[32,1024](offsets, charges, in_mask, out_mask)
    cuda.synchronize()
    return out_mask

def get_in_offsets(offsets, content, indices, mask_rows=None, mask_content=None):
    assert(content.shape == mask_content.shape)
    assert(offsets.shape[0] - 1 == indices.shape[0])
    assert(offsets.shape[0] - 1 == mask_rows.shape[0])
    out = cupy.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    get_in_offsets_cudakernel[32, 1024](offsets, content, indices, mask_rows, mask_content, out)
    cuda.synchronize()
    return out

def set_in_offsets(offsets, content, indices, target, mask_rows=None, mask_content=None):
    assert(content.shape == mask_content.shape)
    assert(offsets.shape[0]-1 == indices.shape[0])
    assert(offsets.shape[0]-1 == target.shape[0])
    assert(offsets.shape[0]-1 == mask_rows.shape[0])
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    set_in_offsets_cudakernel[32, 1024](offsets, content, indices, target, mask_rows, mask_content)
    cuda.synchronize()
                
def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1["eta"].shape)
    assert(mask2.shape == objs2["eta"].shape)
    assert(mask1.shape == objs1["phi"].shape)
    assert(mask2.shape == objs2["phi"].shape)
    assert(objs1["offsets"].shape == objs2["offsets"].shape)
    
    mask_out = cupy.zeros_like(objs1["eta"], dtype=cupy.bool)
    mask_deltar_first_cudakernel[32, 1024](
        objs1["eta"], objs1["phi"], mask1, objs1["offsets"],
        objs2["eta"], objs2["phi"], mask2, objs2["offsets"],
        drcut**2, mask_out
    )
    cuda.synchronize()
    mask_out = cupy.invert(mask_out)
    return mask_out

def histogram_from_vector(data, weights, bins, mask=None):
    assert(len(data) == len(weights))

    allowed_dtypes =[cupy.float32, cupy.int32, cupy.int8] 
    assert(data.dtype in allowed_dtypes)
    assert(weights.dtype in allowed_dtypes)
    assert(bins.dtype in allowed_dtypes)
   
    #Allocate output arrays 
    nblocks = 64
    nthreads = 256
    out_w = cupy.zeros((nblocks, len(bins) - 1), dtype=cupy.float32)
    out_w2 = cupy.zeros((nblocks, len(bins) - 1), dtype=cupy.float32)

    #Fill output
    if len(data) > 0:
        if mask is None:
            fill_histogram[nblocks, nthreads](data, weights, bins, out_w, out_w2)
        else:
            assert(len(data) == len(mask)) 
            fill_histogram_masked[nblocks, nthreads](data, weights, bins, mask, out_w, out_w2)

    cuda.synchronize()

    out_w = out_w.sum(axis=0)
    out_w2 = out_w2.sum(axis=0)

    return cupy.asnumpy(out_w), cupy.asnumpy(out_w2), cupy.asnumpy(bins)

def histogram_from_vector_several(variables, weights, mask):
    all_arrays = []
    all_bins = []
    num_histograms = len(variables)

    for array, bins in variables:
        all_arrays += [array]
        all_bins += [bins]

    max_bins = max([b.shape[0] for b in all_bins])
    stacked_array = cupy.stack(all_arrays, axis=0)
    stacked_bins = cupy.concatenate(all_bins)
    nbins = cupy.array([len(b) for b in all_bins])
    nbins_sum = cupy.cumsum(nbins)
    nbins_sum = cupy.hstack([cupy.array([0]), nbins_sum])

    nblocks = 32
 
    out_w = cupy.zeros((len(variables), nblocks, max_bins), dtype=np.float32)
    out_w2 = cupy.zeros((len(variables), nblocks, max_bins), dtype=np.float32)
    fill_histogram_several[nblocks,256](
        stacked_array, weights, mask, stacked_bins,
        nbins, nbins_sum, out_w, out_w2
    )
    out_w = out_w.sum(axis=1)
    out_w2 = out_w2.sum(axis=1)
    out_w_separated = [cupy.asnumpy(out_w[i, 0:nbins[i]-1]) for i in range(num_histograms)]
    out_w2_separated = [cupy.asnumpy(out_w2[i, 0:nbins[i]-1]) for i in range(num_histograms)]
    
    ret = []
    for ibin in range(len(all_bins)):
        ret += [(out_w_separated[ibin], out_w2_separated[ibin], cupy.asnumpy(all_bins[ibin]))]
    return ret

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_cudakernel[32, 1024](values, edges, contents, out)

def copyto_dst_indices(dst, src, inds_dst):
    assert(len(inds_dst) == len(src))
    copyto_dst_indices_cudakernel[32, 1024](dst, src, inds_dst)
    cuda.synchronize()

def compute_new_offsets(offsets_old, mask_objects, offsets_new):
    assert(len(offsets_old) == len(offsets_new))
    counts = cupy.zeros(len(offsets_old)-1, dtype=np.int32)
    compute_new_offsets_cudakernel[32,1024](offsets_old, mask_objects, counts, offsets_new)
    cuda.synchronize()
