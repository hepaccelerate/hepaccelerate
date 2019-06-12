import sys

try:
    from numba import cuda
    import cupy
except ImportError as e:
    print("Could not import cupy or numba.cuda, hepaccelerate.backend_cuda not usable", file=sys.stderr)
    print("Exception: {0}".format(e.msg), file=sys.stderr)

import math
import numpy as np

@cuda.jit(device=True)
def searchsorted_devfunc(arr, val):
    ret = -1

    #overflow
    if val > arr[-1]:
        return len(arr)

    for i in range(len(arr)):
        if val <= arr[i]:
            ret = i
            break
    return ret

@cuda.jit
def searchsorted_kernel(vals, arr, inds_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(vals), xstride):
        inds_out[i] = searchsorted_devfunc(arr, vals[i])

def searchsorted(arr, vals):
    """
    Find indices to insert vals into arr to preserve order.
    """
    ret = cupy.zeros_like(vals, dtype=cupy.int32)
    searchsorted_kernel[32, 1024](vals, arr, ret)
    return ret 

@cuda.jit
def fill_histogram(data, weights, bins, out_w, out_w2):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for i in range(xi, len(data), xstride):
        bin_idx = searchsorted_devfunc(bins, data[i]) - 1
        if bin_idx >=0 and bin_idx < len(out_w):
            cuda.atomic.add(out_w, bin_idx, weights[i])
            cuda.atomic.add(out_w2, bin_idx, weights[i]**2)

@cuda.jit
def select_opposite_sign_muons_cudakernel(muon_charges_content, muon_charges_offsets, content_mask_in, content_mask_out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    
    for iev in range(xi, muon_charges_offsets.shape[0]-1, xstride):
        start = np.uint64(muon_charges_offsets[iev])
        end = np.uint64(muon_charges_offsets[iev + 1])
        
        ch1 = np.int32(0)
        idx1 = np.uint64(0)
        ch2 = np.int32(0)
        idx2 = np.uint64(0)
        
        for imuon in range(start, end):
            if not content_mask_in[imuon]:
                continue
                
            if idx1 == 0 and idx2 == 0:
                ch1 = muon_charges_content[imuon]
                idx1 = imuon
                continue
            else:
                ch2 = muon_charges_content[imuon]
                if (ch2 != ch1):
                    idx2 = imuon
                    content_mask_out[idx1] = 1
                    content_mask_out[idx2] = 1
                    break
    return

@cuda.jit
def sum_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
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
def max_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
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
def min_in_offsets_cudakernel(content, offsets, mask_rows, mask_content, out):
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
def get_in_offsets_cudakernel(content, offsets, indices, mask_rows, mask_content, out):
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
def set_in_offsets_cudakernel(content, offsets, indices, target, mask_rows, mask_content):
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
        
def sum_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    sum_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=dtype)
    sum_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, sum_offsets)
    cuda.synchronize()
    return sum_offsets

def max_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    max_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def min_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = cupy.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    min_in_offsets_cudakernel[32, 1024](content, struct.offsets, mask_rows, mask_content, max_offsets)
    cuda.synchronize()
    return max_offsets

def select_muons_opposite_sign(muons, in_mask):
    out_mask = cupy.invert(muons.make_mask())
    select_opposite_sign_muons_cudakernel[32,1024](muons.charge, muons.offsets, in_mask, out_mask)
    cuda.synchronize()
    return out_mask

def get_in_offsets(content, offsets, indices, mask_rows, mask_content):
    out = cupy.zeros(len(offsets) - 1, dtype=content.dtype)
    get_in_offsets_cudakernel[32, 1024](content, offsets, indices, mask_rows, mask_content, out)
    cuda.synchronize()
    return out

def set_in_offsets(content, offsets, indices, target, mask_rows, mask_content):
    set_in_offsets_cudakernel[32, 1024](content, offsets, indices, target, mask_rows, mask_content)
    cuda.synchronize()

"""
For all events (N), mask the objects in the first collection (M1) if they are closer than dr2 to any object in the second collection (M2).

    etas1: etas of the first object, array of (M1, )
    phis1: phis of the first object, array of (M1, )
    mask1: mask (enabled) of the first object, array of (M1, )
    offsets1: offsets of the first object, array of (N, )

    etas2: etas of the second object, array of (M2, )
    phis2: phis of the second object, array of (M2, )
    mask2: mask (enabled) of the second object, array of (M2, )
    offsets2: offsets of the second object, array of (N, )
    
    mask_out: output mask, array of (M1, )

"""
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
                dphi = phi1 - phi2 + math.pi
                while dphi > 2*math.pi:
                    dphi -= 2*math.pi
                dphi -= math.pi
                
                #if first object is closer than dr2, mask element will be *disabled*
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr
                
def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1.eta.shape)
    assert(mask2.shape == objs2.eta.shape)
    assert(objs1.offsets.shape == objs2.offsets.shape)
    
    mask_out = cupy.zeros_like(objs1.eta, dtype=cupy.bool)
    mask_deltar_first_cudakernel[32, 1024](
        objs1.eta, objs1.phi, mask1, objs1.offsets,
        objs2.eta, objs2.phi, mask2, objs2.offsets,
        drcut**2, mask_out
    )
    cuda.synchronize()
    mask_out = cupy.invert(mask_out)
    return mask_out

def histogram_from_vector(data, weights, bins):
    assert(len(data) == len(weights))
    out_w = cupy.zeros(len(bins) - 1, dtype=cupy.float32)
    out_w2 = cupy.zeros(len(bins) - 1, dtype=cupy.float32)
    if len(data) > 0:
        fill_histogram[32, 1024](data, weights, bins, out_w, out_w2)
    return cupy.asnumpy(out_w), cupy.asnumpy(out_w2), cupy.asnumpy(bins)


@cuda.jit
def get_bin_contents_cudakernel(values, edges, contents, out):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for i in range(xi, len(values), xstride):
        v = values[i]
        ibin = searchsorted_devfunc(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_cudakernel[32, 1024](values, edges, contents, out)
