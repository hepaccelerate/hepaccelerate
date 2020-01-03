import os
import numba
import numpy as np
import math

@numba.njit(fastmath=True)
def spherical_to_cartesian(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, e

@numba.njit(fastmath=True)
def cartesian_to_spherical(px, py, pz, e):
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    return pt, eta, phi, mass

@numba.njit(fastmath=True)
def add_spherical(pts, etas, phis, masses):
    px_tot = 0.0
    py_tot = 0.0
    pz_tot = 0.0
    e_tot = 0.0
    
    for i in range(len(pts)):
        px, py, pz, e = spherical_to_cartesian(pts[i], etas[i], phis[i], masses[i])
        px_tot += px
        py_tot += py
        pz_tot += pz
        e_tot += e
    return cartesian_to_spherical(px_tot, py_tot, pz_tot, e_tot)

@numba.njit(fastmath=True)
def deltaphi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

"""Returns the first index in arr that is equal or larger than val.

We use this function, rather than np.searchsorted to have a similar implementation
between the CPU and GPU backends.

Args:
    bins: sorted data array
    val: value to find in array

Returns:
    index of first value in array that is equal or larger than val,
    len(bins) otherwise
"""
@numba.njit(fastmath=True)
def searchsorted_devfunc_right(bins, val):
    if val < bins[0]:
        return 0
    if val >= bins[-1]:
        return len(bins) - 1

    ret = np.searchsorted(bins, val, side="right")

    return ret

@numba.njit(fastmath=True)
def searchsorted_devfunc_left(bins, val):
    if val < bins[0]:
        return 0
    if val >= bins[-1]:
        return len(bins) - 1

    ret = np.searchsorted(bins, val, side="left")

    return ret

@numba.njit(parallel=True, fastmath=True)
def searchsorted_right(vals, bins, inds_out):
    for i in numba.prange(len(vals)):
        inds_out[i] = searchsorted_devfunc_right(bins, vals[i])

@numba.njit(parallel=True, fastmath=True)
def searchsorted_left(vals, bins, inds_out):
    for i in numba.prange(len(vals)):
        inds_out[i] = searchsorted_devfunc_left(bins, vals[i])

@numba.njit(fastmath=True, parallel=True)
def fill_histogram_several(data, weights, mask, bins, nbins, nbins_sum, out_w, out_w2):
    #number of histograms to fill 
    ndatavec = data.shape[0]
 
    bin_inds = np.zeros((ndatavec, data.shape[1]), dtype=np.int32)
    for iev in numba.prange(data.shape[1]):
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
                bin_inds[ivec, iev] = bin_idx

    for iev in range(data.shape[1]):
        if mask[iev]:
            for ivec in range(ndatavec):
                bin_idx = bin_inds[ivec, iev]

                if bin_idx >=0 and bin_idx < nbins[ivec]:
                    wi = weights[iev]
                    out_w[ivec, bin_idx] += np.float32(wi)
                    out_w2[ivec, bin_idx] += np.float32(wi**2)

"""Given a data array and weights array, fills a 1D histogram with the weights.

Args:
    data: N-element array of input data
    weights: N-element array of input weights
    bins: sorted bin edges
    out_w: output histogram, filled with weights
    out_w2: output histogram, filled with squared weights for error propagation
"""
@numba.njit(fastmath=True, parallel=True)
def fill_histogram(data, weights, bins, out_w, out_w2):
    assert(len(data) == len(weights))
    assert(len(out_w) == len(out_w2))
    assert(len(bins) - 1 == len(out_w))

    nbins = out_w.shape[0]

    #find the indices of the bins into which every data element would fall
    bin_inds = np.zeros(len(data), dtype=np.int32)
    for i in numba.prange(len(data)):
        bin_idx = searchsorted_devfunc_right(bins, data[i]) - 1
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        bin_inds[i] = bin_idx

    #fill the outputs, cannot parallelize this without atomics
    for i in range(len(data)):
        bin_idx = bin_inds[i]
        wi = weights[i]

        if bin_idx >=0 and bin_idx < len(out_w):
            out_w[bin_idx] += np.float32(wi)
            out_w2[bin_idx] += np.float32(wi**2)

@numba.njit(fastmath=True, parallel=True)
def fill_histogram_masked(data, weights, bins, mask, out_w, out_w2):
    assert(len(data) == len(weights))
    assert(len(out_w) == len(out_w2))
    assert(len(bins) - 1 == len(out_w))

    nbins = out_w.shape[0]

    #find the indices of the bins into which every data element would fall
    bin_inds = np.zeros(len(data), dtype=np.int32)
    for i in numba.prange(len(data)):
        bin_idx = searchsorted_devfunc_right(bins, data[i]) - 1
        if bin_idx >= nbins:
            bin_idx = nbins - 1
        bin_inds[i] = bin_idx


    #fill the outputs, cannot parallelize this without atomics
    for i in range(len(data)):
        if not mask[i]:
            continue

        bin_idx = bin_inds[i]
        wi = weights[i]

        if bin_idx >= nbins:
            bin_idx = nbins - 1

        if bin_idx >=0 and bin_idx < len(out_w):
            out_w[bin_idx] += np.float32(wi)
            out_w2[bin_idx] += np.float32(wi**2)

@numba.njit(parallel=True, fastmath=True)
def select_opposite_sign_kernel(charges_content, charges_offsets, content_mask_in, content_mask_out):
    assert(len(charges_content) == len(content_mask_in))
    assert(len(charges_content) == len(content_mask_out))

    for iev in numba.prange(charges_offsets.shape[0]-1):
        start = np.uint64(charges_offsets[iev])
        end = np.uint64(charges_offsets[iev + 1])
        
        ch1 = np.float32(0.0)
        idx1 = np.uint64(0)
        ch2 = np.float32(0.0)
        idx2 = np.uint64(0)
        
        #loop over objects (e.g. muons)
        for iobj in range(start, end):
            #only consider muons that pass
            if not content_mask_in[iobj]:
                continue
            
            #First object in event
            if idx1 == 0 and idx2 == 0:
                ch1 = charges_content[iobj]
                idx1 = iobj
                continue

            #Rest of the objects
            else:
                ch2 = charges_content[iobj]
                if (ch2 != ch1):
                    idx2 = iobj
                    content_mask_out[idx1] = True
                    content_mask_out[idx2] = True
                    break
    return

@numba.njit(parallel=True, fastmath=True)
def sum_in_offsets_kernel(offsets, content, mask_rows, mask_content, out):
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))
    assert(len(out) == len(offsets) - 1)

    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] += content[ielem]

@numba.njit(parallel=True, fastmath=True)
def prod_in_offsets_kernel(offsets, content, mask_rows, mask_content, out):
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))
    assert(len(out) == len(offsets) - 1)

    for iev in numba.prange(offsets.shape[0]-1):
        if not mask_rows[iev]:
            continue
            
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            if mask_content[ielem]:
                out[iev] *= content[ielem]

@numba.njit(parallel=True, fastmath=True)
def max_in_offsets_kernel(offsets, content, out, mask_rows=None, mask_content=None):
    assert(len(out) == len(offsets) - 1)
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))

    for iev in numba.prange(offsets.shape[0]-1):
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

@numba.njit(parallel=True, fastmath=True)
def min_in_offsets_kernel(offsets, content, mask_rows, mask_content, out):
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))
    assert(len(out) == len(offsets) - 1)
    for iev in numba.prange(offsets.shape[0]-1):
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

@numba.njit(parallel=True, fastmath=True)
def get_in_offsets_kernel(offsets, content, indices, mask_rows, mask_content, out):
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))
    assert(len(offsets) - 1 == len(indices))
    assert(len(out) == len(offsets) - 1)

    for iev in numba.prange(offsets.shape[0] - 1):
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

@numba.njit(parallel=True, fastmath=True)
def set_in_offsets_kernel(offsets, content, indices, target, mask_rows, mask_content):
    for iev in numba.prange(offsets.shape[0] - 1):
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

@numba.njit(parallel=True, fastmath=True)
def broadcast(offsets, content, out):
    assert(offsets.shape[0] - 1 == content.shape[0])
    for iev in numba.prange(offsets.shape[0]-1):
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            out[ielem] = content[iev]
 
@numba.njit(parallel=True, fastmath=True)
def mask_deltar_first_kernel(etas1, phis1, mask1, offsets1, etas2, phis2, mask2, offsets2, dr2, mask_out):
    
    for iev in numba.prange(len(offsets1)-1):
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
                dphi = deltaphi(phi1, phi2)
                
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr

@numba.njit(parallel=True)
def copyto_dst_indices(dst, src, inds_dst):
    assert(len(inds_dst) == len(src))
    for i1 in numba.prange(len(src)):
        i2 = inds_dst[i1] 
        dst[i2] = src[i1]

    
@numba.njit(parallel=True, fastmath=True)
def get_bin_contents_kernel(values, edges, contents, out):
    for i in numba.prange(len(values)):
        v = values[i]
        ibin = searchsorted_devfunc_right(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

@numba.njit(parallel=True, fastmath=True)
def apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out):
    for iev in numba.prange(len(runs)):
        run = runs[iev]
        lumi = lumis[iev]

        if run in masks:
            lumimask = masks[run]
            ind = searchsorted_devfunc_right(lumimask, lumi)
            if np.mod(ind, 2) == 1:
                mask_out[iev] = 1

@numba.njit(parallel=True, fastmath=True)
def compute_new_offsets(offsets_old, mask_objects, offsets_new):
    counts = np.zeros(len(offsets_old) - 1, dtype=np.int64)
    for iev in numba.prange(len(offsets_old) - 1):
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

def make_masks(offsets, content, mask_rows, mask_content):
    if mask_rows is None:
        mask_rows = np.ones(len(offsets) - 1, dtype=np.bool)
    if mask_content is None:
        mask_content = np.ones(len(content), dtype=np.bool)
    return mask_rows, mask_content

#User-friendly functions that call the kernels, but create the output arrays themselves
def sum_in_offsets(offsets, content, mask_rows=None, mask_content=None, dtype=None):
    if not dtype:
        dtype = content.dtype
    res = np.zeros(len(offsets) - 1, dtype=dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    sum_in_offsets_kernel(offsets, content, mask_rows, mask_content, res)
    return res

def prod_in_offsets(offsets, content, mask_rows=None, mask_content=None, dtype=None):
    if not dtype:
        dtype = content.dtype
    res = np.ones(len(offsets) - 1, dtype=dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    prod_in_offsets_kernel(offsets, content, mask_rows, mask_content, res)
    return res

def max_in_offsets(offsets, content, mask_rows=None, mask_content=None):
    max_offsets = np.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    max_in_offsets_kernel(offsets, content, max_offsets, mask_rows, mask_content)
    return max_offsets

def min_in_offsets(offsets, content, mask_rows=None, mask_content=None):
    max_offsets = np.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    min_in_offsets_kernel(offsets, content, mask_rows, mask_content, max_offsets)
    return max_offsets

def select_opposite_sign(offsets, charges, in_mask):
    out_mask = np.zeros(len(charges), dtype=np.bool)
    select_opposite_sign_kernel(charges, offsets, in_mask, out_mask)
    return out_mask

def get_in_offsets(offsets, content, indices, mask_rows=None, mask_content=None):
    out = np.zeros(len(offsets) - 1, dtype=content.dtype)
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    get_in_offsets_kernel(offsets, content, indices, mask_rows, mask_content, out)
    return out

def set_in_offsets(offsets, content, indices, target, mask_rows=None, mask_content=None):
    mask_rows, mask_content = make_masks(offsets, content, mask_rows, mask_content) 
    set_in_offsets_kernel(offsets, content, indices, target, mask_rows, mask_content)

def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1["eta"].shape)
    assert(mask2.shape == objs2["eta"].shape)
    assert(mask1.shape == objs1["phi"].shape)
    assert(mask2.shape == objs2["phi"].shape)
    assert(objs1["offsets"].shape == objs2["offsets"].shape)
    
    mask_out = np.zeros_like(objs1["eta"], dtype=np.bool)
    mask_deltar_first_kernel(
        objs1["eta"], objs1["phi"], mask1, objs1["offsets"],
        objs2["eta"], objs2["phi"], mask2, objs2["offsets"],
        drcut**2, mask_out
    )
    mask_out = np.invert(mask_out)
    return mask_out

def histogram_from_vector_several(variables, weights, mask):
    all_arrays = []
    all_bins = []
    num_histograms = len(variables)

    for array, bins in variables:
        all_arrays += [array]
        all_bins += [bins]

    for a in all_arrays:
        assert(a.shape == all_arrays[0].shape)
        assert(a.shape == weights.shape)
        assert(a.shape == mask.shape)

    max_bins = max([b.shape[0] for b in all_bins])
    stacked_array = np.stack(all_arrays, axis=0)
    stacked_bins = np.concatenate(all_bins)
    nbins = np.array([len(b) for b in all_bins])
    nbins_sum = np.cumsum(nbins)
    nbins_sum = np.insert(nbins_sum, 0, [0])
    
    out_w = np.zeros((len(variables), max_bins), dtype=np.float32)
    out_w2 = np.zeros((len(variables), max_bins), dtype=np.float32)
    fill_histogram_several(
        stacked_array, weights, mask, stacked_bins,
        nbins, nbins_sum, out_w, out_w2
    )
    out_w_separated = [out_w[i, 0:nbins[i]-1] for i in range(num_histograms)]
    out_w2_separated = [out_w2[i, 0:nbins[i]-1] for i in range(num_histograms)]

    ret = []
    for ibin in range(len(all_bins)):
        ret += [(out_w_separated[ibin], out_w2_separated[ibin], all_bins[ibin])]
    return ret

def histogram_from_vector(data, weights, bins, mask=None):
    assert(len(data) == len(weights))
    out_w = np.zeros(len(bins) - 1, dtype=np.float64)
    out_w2 = np.zeros(len(bins) - 1, dtype=np.float64)
    if mask is None:
        fill_histogram(data, weights, bins, out_w, out_w2) 
    else:
        fill_histogram_masked(data, weights, bins, mask, out_w, out_w2) 
    return out_w, out_w2, bins

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_kernel(values, edges, contents, out)

def searchsorted(bins, vals, side="left"):
    out = np.zeros(len(vals), dtype=np.int32)
    if side == "left":
        searchsorted_left(vals, bins, out)
    else:
        searchsorted_right(vals, bins, out)
    return out
