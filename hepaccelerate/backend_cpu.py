import os
import numba
import numpy as np
import math

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

def searchsorted(bins, vals, side):
    out = np.zeros(len(vals), dtype=np.int32)
    if side == "left":
        searchsorted_left(vals, bins, out)
    else:
        searchsorted_right(vals, bins, out)
    return out


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

"""Given a vector of muon charges and event offsets, masks the first two opposite sign muons.

Args:
    muon_charges_content: Array of muon charges (N-elem)
    muon_charges_offsets: Array of event offsets
    content_mask_in: Mask of muons to be used for this kernel (N-elem)
    content_mask_out: Mask where the passing muons will be set to True (N-elem)
"""
@numba.njit(parallel=True, fastmath=True)
def select_opposite_sign_muons_kernel(muon_charges_content, muon_charges_offsets, content_mask_in, content_mask_out):
    assert(len(muon_charges_content) == len(content_mask_in))
    assert(len(muon_charges_content) == len(content_mask_out))

    for iev in numba.prange(muon_charges_offsets.shape[0]-1):
        start = np.uint64(muon_charges_offsets[iev])
        end = np.uint64(muon_charges_offsets[iev + 1])
        
        ch1 = np.float32(0.0)
        idx1 = np.uint64(0)
        ch2 = np.float32(0.0)
        idx2 = np.uint64(0)
        
        for imuon in range(start, end):
            #only consider muons that pass
            if not content_mask_in[imuon]:
                continue
            
            #First muon in event
            if idx1 == 0 and idx2 == 0:
                ch1 = muon_charges_content[imuon]
                idx1 = imuon
                continue

            #Next mouns
            else:
                ch2 = muon_charges_content[imuon]
                if (ch2 != ch1):
                    idx2 = imuon
                    content_mask_out[idx1] = True
                    content_mask_out[idx2] = True
                    break
    return

"""Sums a content array within event offsets, taking into account masks.

Args:
    content: data array, N elements
    offsets: event offset array, M+1 elements
    mask_rows: events/rows to consider, M elements
    mask_content: data elements to consider, N elements 
    out: output array, M elements
"""
@numba.njit(parallel=True, fastmath=True)
def sum_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):
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
def prod_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):
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

"""Finds the maximum value of a content array within events

Args:
    content: input data array, N elements
    offsets: event offset array, M+1 elements
    mask_rows: events/rows to consider, M elements
    mask_content: data elements to consider, N elements 
    out: output array, M elements
"""
@numba.njit(parallel=True, fastmath=True)
def max_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):
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
                if first or content[ielem] > accum:
                    accum = content[ielem]
                    first = False
        out[iev] = accum

"""Finds the minimum value of a content array within events

Args:
    content: input data array, N elements
    offsets: event offset array, M+1 elements
    mask_rows: events/rows to consider, M elements
    mask_content: data elements to consider, N elements 
    out: output array, M elements
"""
@numba.njit(parallel=True, fastmath=True)
def min_in_offsets_kernel(content, offsets, mask_rows, mask_content, out):
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

"""Retrieves the n-th value of a data array in an event given offsets, where n is an index.

Can be used to retrieve e.g. the pT of the second jet from each event into a contiguous array.

Args:
    content: input data array, N elements
    offsets: event offset array, M+1 elements
    indices: indices to retrieve from events, M elements
    mask_rows: events/rows to consider, M elements
    mask_content: data elements to consider, N elements 
    out: output array, M elements
"""
@numba.njit(parallel=True, fastmath=True)
def get_in_offsets_kernel(content, offsets, indices, mask_rows, mask_content, out):
    assert(len(content) == len(mask_content))
    assert(len(offsets) - 1 == len(mask_rows))
    assert(len(offsets) - 1 == len(indices))
    assert(len(out) == len(offsets) - 1)

    for iev in numba.prange(offsets.shape[0]-1):
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

"""Sets the n-th value of a data array in an event given offsets, where n is an index.

Can be used to set e.g. the pT of the second jet in each event to a value from a contiguous array.

Args:
    content: input data array, N elements
    offsets: event offset array, M+1 elements
    indices: indices to set in events, M elements
    target: target values to set in events, M elements
    mask_rows: events/rows to consider, M elements
    mask_content: data elements to consider, N elements 
"""
@numba.njit(parallel=True, fastmath=True)
def set_in_offsets_kernel(content, offsets, indices, target, mask_rows, mask_content):
    for iev in numba.prange(offsets.shape[0]-1):
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
def broadcast(content, offsets, out):
    for iev in numba.prange(offsets.shape[0]-1):
        start = offsets[iev]
        end = offsets[iev + 1]
        for ielem in range(start, end):
            out[ielem] = content[iev]
 
""" For all events (N), mask the objects in the first collection (M1)
  if closer than dr2 to any object in the second collection (M2).

Args:
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
                dphi = np.mod(phi1 - phi2 + math.pi, 2*math.pi) - math.pi
                
                passdr = ((deta**2 + dphi**2) < dr2)
                mask_out[idx1] = mask_out[idx1] | passdr

@numba.njit(parallel=True)
def copyto_dst_indices(dst, src, inds_dst):
    assert(len(inds_dst) == len(src))
    for i1 in numba.prange(len(src)):
        i2 = inds_dst[i1] 
        dst[i2] = src[i1]

#User-friendly functions that call the kernels, but create the output arrays themselves

def sum_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    res = np.zeros(len(struct.offsets) - 1, dtype=dtype)
    sum_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, res)
    return res

def prod_in_offsets(struct, content, mask_rows, mask_content, dtype=None):
    if not dtype:
        dtype = content.dtype
    res = np.ones(len(struct.offsets) - 1, dtype=dtype)
    prod_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, res)
    return res

def max_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = np.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    max_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, max_offsets)
    return max_offsets

def min_in_offsets(struct, content, mask_rows, mask_content):
    max_offsets = np.zeros(len(struct.offsets) - 1, dtype=content.dtype)
    min_in_offsets_kernel(content, struct.offsets, mask_rows, mask_content, max_offsets)
    return max_offsets

def select_muons_opposite_sign(muons, in_mask):
    out_mask = np.invert(muons.make_mask())
    select_opposite_sign_muons_kernel(muons.charge, muons.offsets, in_mask, out_mask)
    return out_mask

def get_in_offsets(content, offsets, indices, mask_rows, mask_content):
    out = np.zeros(len(offsets) - 1, dtype=content.dtype)
    get_in_offsets_kernel(content, offsets, indices, mask_rows, mask_content, out)
    return out

def set_in_offsets(content, offsets, indices, target, mask_rows, mask_content):
    set_in_offsets_kernel(content, offsets, indices, target, mask_rows, mask_content)
                
def mask_deltar_first(objs1, mask1, objs2, mask2, drcut):
    assert(mask1.shape == objs1.eta.shape)
    assert(mask2.shape == objs2.eta.shape)
    assert(objs1.offsets.shape == objs2.offsets.shape)
    
    mask_out = np.zeros_like(objs1.eta, dtype=np.bool)
    mask_deltar_first_kernel(
        objs1.eta, objs1.phi, mask1, objs1.offsets,
        objs2.eta, objs2.phi, mask2, objs2.offsets,
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
    return out_w_separated, out_w2_separated, all_bins

def histogram_from_vector(data, weights, bins, mask=None):
    assert(len(data) == len(weights))
    out_w = np.zeros(len(bins) - 1, dtype=np.float64)
    out_w2 = np.zeros(len(bins) - 1, dtype=np.float64)
    if mask is None:
        fill_histogram(data, weights, bins, out_w, out_w2) 
    else:
        fill_histogram_masked(data, weights, bins, mask, out_w, out_w2) 
    return out_w, out_w2, bins
    
@numba.njit(parallel=True, fastmath=True)
def get_bin_contents_kernel(values, edges, contents, out):
    for i in numba.prange(len(values)):
        v = values[i]
        ibin = searchsorted_devfunc_right(edges, v)
        if ibin>=0 and ibin < len(contents):
            out[i] = contents[ibin]

def get_bin_contents(values, edges, contents, out):
    assert(values.shape == out.shape)
    assert(edges.shape[0] == contents.shape[0]+1)
    get_bin_contents_kernel(values, edges, contents, out)


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
    counts = np.zeros(len(offsets_old)-1, dtype=np.int64)
    for iev in numba.prange(len(offsets_old)-1):
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
