"""This file contains the public-facing API of the kernels
"""

def spherical_to_cartesian(backend, pt, eta, phi, mass):
    """Converts an array of spherical four-momentum coordinates (pt, eta, phi, mass) to cartesian (px, py ,pz, E).
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        pt (array of floats): Data array of the transverse momentum values (numpy or cupy)
        eta (array of floats): Data array of the pseudorapidity
        phi (array of floats): Data array of the azimuthal angle
        mass (array of floats): Data array of the mass
    
    Returns:
        tuple of arrays: returns the numpy or cupy arrays (px, py, pz, E) 
    """
    return backend.spherical_to_cartesian(pt, eta, phi, mass)

def cartesian_to_spherical(backend, px, py, pz, e):
    """Converts an array of cartesian four-momentum coordinates (px, py ,pz, E) to spherical (pt, eta, phi, mass).
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        px (array of floats): Data array of the momentum x coordinate (numpy or cupy)
        py (array of floats): Data array of the momentum y coordinate
        pz (array of floats): Data array of the momentum z coordinate
        e (array of floats): Data array of the energy values
    
    Returns:
        tuple of arrays: returns the numpy or cupy arrays (pt, eta, phi, mass)
    """
    return backend.cartesian_to_spherical(px, py, pz, e)

def searchsorted(backend, arr, vals, side="right"):
    """Finds where to insert the values in 'vals' into a sorted array 'arr' to preserve order, as np.searchsorted.
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        arr (array of floats): sorted array of bin edges
        vals (array of floats): array of values to insert into the sorted array
        side (str, optional): "left" or "right" as in np.searchsorted
    
    Returns:
        array of ints: Indices into 'arr' where the values would be inserted 
    """
    return backend.searchsorted(arr, vals, side=side)

def broadcast(backend, offsets, content, out):
    """Given the offsets from a one-dimensional jagged array, broadcasts a per-event array to a per-object array.

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0]])
    >>> inp = numpy.array([123.0, 456.0])
    >>> out = numpy.zeros_like(j.content)
    >>> broadcast(backend_cpu, j.offsets, inp, out)
    >>> j2 = awkward.JaggedArray.fromoffsets(j.offsets, out)
    >>> r = (j2 == awkward.fromiter([[123.0, 123.0],[456.0, 456.0, 456.0]]))
    >>> assert(numpy.all(r.content))
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (array of uint64): one dimensional offsets of the jagged array (depth 1)
        content (array of floats): per-event array of inputs to broadcast
        out (array of floats): per-element output
    """
    backend.broadcast(offsets, content, out)

def sum_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None, dtype=None):
    """Sums the values in a depth-1 jagged array within the offsets, e.g. to compute a per-event sum
    
    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = sum_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([3.0, 7.0, 13.0, 0.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (array of uint64): one dimensional offsets of the jagged array (depth 1)
        content (array): data array to sum over
        mask_rows (array of bool, optional): Mask the values in the offset array that are set to False
        mask_content (array of bool, optional): Mask the values in the data array that are set to False
        dtype (data type, optional): Output data type, useful to specify e.g. int8 when summing over booleans
    
    Returns:
        array: Totals within the offsets
    """
    return backend.sum_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content, dtype=dtype)

def prod_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None, dtype=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = prod_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([2.0, 12.0, 42.0, 1.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (TYPE): Description
        mask_content (TYPE): Description
        dtype (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.prod_in_offsets(offsets, content, mask_rows, mask_content, dtype=dtype)

def max_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = max_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([2.0, 4.0, 7.0, 0.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.max_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def min_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = min_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([1.0, 3.0, 6.0, 0.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.min_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def select_opposite_sign(backend, offsets, charges, in_mask):
    """Summary
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        charges (TYPE): Description
        in_mask (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.select_opposite_sign(offsets, charges, in_mask)

def get_in_offsets(backend, offsets, content, indices, mask_rows=None, mask_content=None):
    """Retrieves the per-event values corresponding to indices in the content array.

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> #Retrieve the first non-masked value in the first and second event, and the second value in the third and fourth event
    >>> inds = numpy.array([0, 0, 1, 1])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, False, True, True, True, True, True]) # Disable the 3rd value (3.0)
    >>> r = get_in_offsets(backend_cpu, j.offsets, j.content, inds, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([1.0, 4.0, 7.0, 0.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        content (TYPE): Description
        indices (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.get_in_offsets(offsets, content, indices, mask_rows=mask_rows, mask_content=mask_content) 

def set_in_offsets(backend, offsets, content, indices, target, mask_rows=None, mask_content=None):
    """Sets the per-event values corresponding to indices in the content array to the values in the target array.
    
    >>> j = awkward.fromiter([[0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0], [0.0]])
    >>> inds = numpy.array([0, 0, 1, 1])
    >>> target = numpy.array([1, 2, 3, 4])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event
    >>> mc = numpy.array([True, True, False, True, True, True, True, True]) # Disable the 3rd value in the content array
    >>> set_in_offsets(backend_cpu, j.offsets, j.content, inds, target, mask_rows=mr, mask_content=mc)
    >>> r = j == awkward.fromiter([[1.0, 0.0],[0.0, 2.0, 0.0], [0.0, 3.0], [0.0]])
    >>> assert(numpy.all(r.content))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (TYPE): Description
        content (TYPE): Description
        indices (TYPE): Description
        target (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    """
    backend.set_in_offsets(offsets, content, indices, target, mask_rows=mask_rows, mask_content=mask_content)
 
def mask_deltar_first(backend, objs1, mask1, objs2, mask2, drcut):
    """Masks objects in the first collection that are closer than drcut to
       objects in the second collection according to dR=sqrt(dEta^2 + dPhi^2) 

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        objs1 (dict): a dictionary containing the "eta" and "phi" arrays for the first collection
        mask1 (array of bool): Mask of objects in the first collection that are not used for the matching
        objs2 (TYPE): a dictionary containing the "eta" and "phi" arrays for the second collection
        mask2 (array of bool): Mask of objects in the second collection that are not used for the matching
        drcut (float): Minimum delta R value between objects
    
    Returns:
        array of bool: Mask for objects in the first collection that are closer
            than drcut to objects in the second collection
    """
    return backend.mask_deltar_first(objs1, mask1, objs2, mask2, drcut)

def histogram_from_vector(backend, data, weights, bins, mask=None):
    """Fills the weighted values in a data array to a histogram specified by a sorted one-dimensional bin array.

    >>> data = numpy.array([1,1,1,2,3], dtype=numpy.float32)
    >>> weights = numpy.array([1,1,1,2,1], dtype=numpy.float32)
    >>> bins = numpy.array([0,1,2,3,4,5], dtype=numpy.float32)
    >>> mask = numpy.array([False, True, True, True, True])
    >>> r = histogram_from_vector(backend_cpu, data, weights, bins, mask=mask)
    >>> assert(numpy.all(r[0] == numpy.array([0., 2., 2., 1., 0.])))
    >>> assert(numpy.all(r[1] == numpy.array([0., 2., 4., 1., 0.])))
    >>> assert(numpy.all(r[2] == bins))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        data (array of floats): Data array of samples to fill into the histogram
        weights (array of floats): Per-sample weights
        bins (array of floats): Sorted one dimensional bin array
        mask (array of bool, optional): Sample mask to disable filling certain elements
    
    Returns:
        tuple of arrays (w, w^2, bins): A tuple of weight, squared weight and bin arrays
    """
    return backend.histogram_from_vector(data, weights, bins, mask=mask)
 
def histogram_from_vector_several(backend, variables, weights, mask):
    """Fills several data arrays into histograms simultaneously. On a GPU, this is
        faster than calling the histogram function several times due to the overhead
        of calling simple kernels.

    >>> variables = [(numpy.array([1,1,1,2,3], dtype=numpy.float32), numpy.array([0,1,2,3,4,5], dtype=numpy.float32))]
    >>> weights = numpy.array([1,1,1,2,1], dtype=numpy.float32)
    >>> mask = numpy.array([False, True, True, True, True])
    >>> r = histogram_from_vector_several(backend_cpu, variables, weights, mask)
    >>> assert(numpy.all(r[0][0] == numpy.array([0., 2., 2., 1., 0.])))
    >>> assert(numpy.all(r[0][1] == numpy.array([0., 2., 4., 1., 0.])))
    >>> assert(numpy.all(r[0][2] == variables[0][1]))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        variables (list of (data, bins) tuples): Pairs of sample data and bin arrays
        weights (array of floats): Per-sample weight array
        mask (array of bool): Per-sample mask to enable or disable samples
    
    Returns:
        List of (w, w^2, bins) tuples: A list of tuples of weight, squared weight and bin arrays for each variable
    """
    return backend.histogram_from_vector_several(variables, weights, mask) 

def get_bin_contents(backend, values, edges, contents, out):
    """Does a lookup on the values given a set of sorted edges and contents forming a histogram.

    >>> values = numpy.array([1,1,2,3])
    >>> edges = numpy.array([1,2,3,4,5])
    >>> contents = numpy.array([10,20,30,40])
    >>> out = numpy.zeros_like(values)
    >>> get_bin_contents(backend_cpu, values, edges, contents, out)
    >>> assert(numpy.all(out == numpy.array([20, 20, 30, 40])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        values (TYPE): Description
        edges (TYPE): Description
        contents (TYPE): Description
        out (TYPE): Description
    """
    backend.get_bin_contents(values, edges, contents, out) 

def copyto_dst_indices(backend, dst, src, inds_dst):
    """Copies the values from the src array to the specified indices in the dst array

    >>> src = numpy.array([1,2,3])
    >>> dst = numpy.array([0,0,0])
    >>> inds = numpy.array([2,1,0])
    >>> copyto_dst_indices(backend_cpu, dst, src, inds)
    >>> assert(numpy.all([3,2,1]))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        dst (TYPE): Description
        src (TYPE): Description
        inds_dst (TYPE): Description
    """
    backend.copyto_dst_indices(dst, src, inds_dst) 

def compute_new_offsets(backend, offsets_old, mask_objects, offsets_new):
    """Masks elements in a jagged array, creating a new offset array

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0]])
    >>> mask = numpy.array([True, False, True, False, True])
    >>> o = numpy.zeros_like(j.offsets)
    >>> compute_new_offsets(backend_cpu, j.offsets, mask, o)
    >>> j2 = awkward.JaggedArray.fromoffsets(o, j.content[mask])
    >>> j3 = awkward.fromiter([[1.0],[3.0, 5.0]])
    >>> assert(numpy.all(j2.content == j3.content))
    >>> assert(numpy.all(j2.offsets == j3.offsets))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets_old (TYPE): Description
        mask_objects (TYPE): Description
        offsets_new (TYPE): Description
    """
    backend.compute_new_offsets(offsets_old, mask_objects, offsets_new)

if __name__ == "__main__":
    import doctest
    import awkward, numpy
    import hepaccelerate.backend_cpu as backend_cpu
    doctest.testmod()
