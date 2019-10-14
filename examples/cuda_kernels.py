import math
import numba
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def set_array(arr, pt, eta, phi, mass, i):
    arr[0] = pt[i]
    arr[1] = eta[i]
    arr[2] = phi[i]
    arr[3] = mass[i]

@cuda.jit(device=True)
def spherical_to_cartesian(p4):
    pt = p4[0]
    eta = p4[1]
    phi = p4[2]
    mass = p4[3]

    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    p4[0] = px
    p4[1] = py
    p4[2] = pz
    p4[3] = e

@cuda.jit(device=True)
def inv_mass_3(p1c, p2c, p3c):
    px = p1c[0] + p2c[0] + p3c[0]
    py = p1c[1] + p2c[1] + p3c[1]
    pz = p1c[2] + p2c[2] + p3c[2]
    e = p1c[3] + p2c[3] + p3c[3]
    inv_mass = math.sqrt(-(px**2 + py**2 + pz**2 - e**2))
    return inv_mass

@cuda.jit
def comb_3_invmass_closest(pt, eta, phi, mass, offsets, candidate_mass, out_mass, out_best_comb):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    #need to allocate a fixed size array
    maxobj = 100

    for iev in range(xi, offsets.shape[0]-1, xstride):
        start = offsets[iev]
        end = offsets[iev + 1]
        nobj = end - start
        assert(nobj < maxobj)

        #create a arrays of the cartesian components
        p = cuda.local.array((maxobj, 4), numba.float32)
        for iobj in range(start, end):
            set_array(p[iobj - start, :], pt, eta, phi, mass, iobj)
            spherical_to_cartesian(p[iobj - start, :])

        #mass delta R to previous
        delta_previous = 1e10
        iobj_best = 0
        jobj_best = 0
        kobj_best = 0
        mass_best = 0

        #compute the invariant mass of all combinations
        for iobj in range(start, end):
            for jobj in range(iobj + 1, end):
                for kobj in range(jobj + 1, end):
                    inv_mass = inv_mass_3(p[iobj - start, :], p[jobj - start, :], p[kobj - start, :])
                    delta = abs(inv_mass - candidate_mass)
                    if delta < delta_previous:
                        mass_best = inv_mass
                        iobj_best = iobj
                        jobj_best = jobj
                        kobj_best = kobj
                        delta_previous = delta

        out_mass[iev] = mass_best
        out_best_comb[iev, 0] = iobj_best - start
        out_best_comb[iev, 1] = jobj_best - start
        out_best_comb[iev, 2] = kobj_best - start

@cuda.jit(device=True)
def max_arr(arr):
    m = arr[0]
    for i in range(len(arr)):
        if arr[i] > m:
            m = arr[i]
    return m
    
@cuda.jit
def max_val_comb(vals, offsets, best_comb, out_vals):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)

    ncomb = 3

    for iev in range(xi, offsets.shape[0]-1, xstride):
        start = offsets[iev]
        end = offsets[iev + 1]

        vals_comb = cuda.local.array(ncomb, numba.float32)
        for icomb in range(ncomb):
            idx_jet = best_comb[iev, icomb]
            if idx_jet >= 0:
                vals_comb[icomb] = vals[start + idx_jet]

        out_vals[iev] = max_arr(vals_comb)