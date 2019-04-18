#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from math import ceil
import pickle

from sympy import *
from sympy.polys.groebnertools import *

import numpy as np
from numba import cuda

# Multiplicative Inverses
mi_65521 = open('./mul_inv_65521', 'rb')
MI_65521 = pickle.load(mi_65521)
mi_65521.close()


def cp_cuda(p1, p2, ring):
    """
    Initiates Critical Pair computation on GPU.
    Called in the same way as critical_pair, but
    has many supporting functions.

    The cuda arrays will all be of type np.float32
    because that's what they tell me to do. 
    Will require casting on parse depending on domain

    Prepares data to send to PyCUDA/Numba Cuda.Jit kernel.
    
    Input: p1, p2 : the labeled polynomials B[i], B[j]
           ring   : just passing through to parser
    """
    mod_pair = modified_pair((p1, p2))

    modulus = ring.domain.mod
    nvars = len(ring.symbols)  # for striding
    lt_buf = np.zeros(nvars + 1, dtype=np.int32)
    lt_buf[-1] = 1
    fdest = np.zeros(2 * nvars + 1, dtype=np.int32)
    gdest = np.zeros_like(fdest)

    f = get_cuda_cp_array(mod_pair[0], nvars)
    g = get_cuda_cp_array(mod_pair[1], nvars)

    kernel_data = [nvars, modulus, lt_buf, f, g, fdest, gdest]
    cuda_cp_arys = numba_cp_kernel_launch(kernel_data)

    # After kernel, append modular inverse to fdest gdest [-1]
    if (f[-1] % modulus) == 1:
        fdest[-1] = 1
    else:
        fdest[-1] = MI_32003[str(f[-1] % modulus)]
    if (g[-1] % modulus) == 1:
        gdest[-1] = 1
    else:
        gdest[-1] = MI_32003[str(g[-1] % modulus)]
        
    
    gpu_cp = parse_cuda_cp_to_sympy(cuda_cp_arys, (p1, p2), ring)

    return gpu_cp


def modified_pair(pair):
    """
    Returns truncated lbp pair for transformation
    to cuda arrays.

    Input: pair : a tuple of two labeled polynomials
    Output: modified_pair : a tuple of the components
                            we operate on in critical
                            pair computation Sign(f/g r)
                            multiplier and leading terms
                            of f and g.
    """
    modified_pair = []
    p1 = []
    p2 = []

    # Only get signature multiplier and LT
    p1.append(pair[0][0][0])
    p1.append(Polyn(pair[0]).LT)
    modified_pair.append(tuple(p1))

    p2.append(pair[1][0][0])
    p2.append(Polyn(pair[1]).LT)
    modified_pair.append(tuple(p2))

    return tuple(modified_pair)


def get_cuda_cp_array(mod_pair_part, nvars):
    """
    Fills a np.array for cuda cp with data
    appropriate for CP calculation
    Modifies in place.
    """
    cuda_cp_array = np.zeros(2 * nvars + 1, dtype=np.int32)

    # Signature Multiplier
    for i, s in enumerate(mod_pair_part[0]):
        cuda_cp_array[i] = s

    # Leading Monomial
    for i, e in enumerate(mod_pair_part[1][0]):
        cuda_cp_array[i + nvars] = e

    # Leading Coefficient
    cuda_cp_array[-1] = mod_pair_part[1][1]

    return cuda_cp_array

def numba_cp_kernel_launch(kernel_data):
    """
    Prepared nparray data for numba cuda jit
    Appears to modify sent arrays in place from
    their documentation
    """
    nvars = kernel_data[0]
    modulus = kernel_data[1]
    lt_buf = kernel_data[2]
    f = kernel_data[3]
    g = kernel_data[4]
    fdest = kernel_data[5]
    gdest = kernel_data[6]

    # Launch kernel
    tpb = 32
    bpg = (fdest.size + (tpb - 1)) // tpb

    numba_critical_pairs[tpb, bpg](nvars, modulus, lt_buf, f, g, fdest, gdest)

    return [fdest, gdest]


@cuda.jit
def numba_critical_pairs(nvars, modulus, lt_buf, f, g, fdest, gdest):
    """
    Numba Cuda.Jit kernel for critical pair computation.

    INPUT:
    nvars: integer, used as array stride

    lt_buf: intermediate storage for monomial_lcm(f, g) (len nvars + 1)
               lt_buf[:nvars] : monomial
               lt_buf[-1] : 1 (ring.domain.one)
        * This and [f:g]dest should be in the shared memory for all threads
          after computation

    f, g : polynomials for cp computation of len 2*nvars + 1
           f[0:nvars] : signature multiplier field
           f[nvars:-1] : leading monomial of f
           f[-1] : leading coefficient of f

    fdest : a destination array for final result
            fdest[:nvars] : sig(fr) multiplier field
            fdest[nvars:2*nvars+1] : um field
            fdest[nvars:2*nvars] : um monomial field
            fdest[2*nvars] : um coefficient
            same for g

    OUTPUT: fdest, gdest arrays appropriately filled.

    Procedure:
    1) Compute lt: max of f[i], g[i] for i in range(nvars, 2*nvars+1)
       (the lt is initialized with 1 as its last entry, so we're good there)
    2) Synchronize Threads
    3) dest, lt should be put into shared memory

    4) Compute um and vm simultaneously (no data dependency)
       subtraction for first nvars, division for last entry
       um, vm are stored in their respective fields in dest
    5) Synchronize threads

    6) Compute sig(fr) mult, sig(gr) mult simultaneously (no dependency)
       sum of respective sig in f or g, um or vm fields in dest sig fields.
    7) Synchronize threads
    8) Copy fdest, gdest back to host
    """    
    i = cuda.grid(1)
    # lt
    if i < lt_buf.size - 1:
        lt_buf[i] = max((f[nvars + i] % modulus), (g[nvars + i] % modulus))

    cuda.syncthreads()
    # um vm
    if i < lt_buf.size - 1:
        fdest[nvars + i] = ((lt_buf[i] % modulus) - (f[nvars + i] % modulus)) % modulus
        gdest[nvars + i] = ((lt_buf[i] % modulus) - (g[nvars + i] % modulus)) % modulus

    cuda.syncthreads()
    # sig fr gr
    if i < lt_buf.size - 1:
        fdest[i] = ((f[i] % modulus) + (fdest[nvars + i] % modulus)) % modulus
        gdest[i] = ((g[i] % modulus) + (gdest[nvars + i] % modulus)) % modulus


def parse_cuda_cp_to_sympy(cuda_cp, pair, ring):
    """
    Convert cuda_cp array to sympy's 6-tuple form
    by passing through the parts of pair that are
    unmodified during cp computation

    Input: cuda_cp : a list of 2 numpy arrays with
                     Sign(fr) multiplier in [0:nvars]
                     um [nvars:end] with um coefficient
                     at cuda_cp[-1]

           pair: two labeled polynomials from B
                 indices match cuda_cp arrays

           ring: need it for domain
    """
    nvars = len(ring.symbols)

    gpu_sympy_cp = []

    # Build critical pair list
    gpu_sympy_cp.append([cuda_cp[0][:nvars], pair[0][0][1]])  # sig(fr)
    gpu_sympy_cp.append([cuda_cp[0][nvars:-1], cuda_cp[0][-1]])  # um
    gpu_sympy_cp.append(pair[0])  # f
    gpu_sympy_cp.append([cuda_cp[1][:nvars], pair[1][0][1]])  # sig(gr)
    gpu_sympy_cp.append([cuda_cp[1][nvars:-1], cuda_cp[1][-1]])  # vm
    gpu_sympy_cp.append(pair[1])  # g

    # Retuple
    gpu_sympy_cp[0][0] = tuple([s for s in gpu_sympy_cp[0][0]])
    gpu_sympy_cp[1][0] = tuple([e for e in gpu_sympy_cp[1][0]])
    gpu_sympy_cp[3][0] = tuple([s for s in gpu_sympy_cp[3][0]])
    gpu_sympy_cp[4][0] = tuple([e for e in gpu_sympy_cp[4][0]])

    gpu_sympy_cp[0] = tuple(gpu_sympy_cp[0])
    gpu_sympy_cp[1] = tuple(gpu_sympy_cp[1])
    gpu_sympy_cp[3] = tuple(gpu_sympy_cp[3])
    gpu_sympy_cp[4] = tuple(gpu_sympy_cp[4])

    return tuple(gpu_sympy_cp)


if __name__ == "__main__":
    print("GPU Critical Pairs Test")
    print("Cyclic Affine 4")
    
    r, a, b, c, d = ring(symbols="a, b, c, d", domain=GF(65521), order="grevlex")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    F = [f1, f2, f3, f4]

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]
    CP = [critical_pair(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    GPU_CP = [cp_cuda(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    
    for gcp, cp in zip(GPU_CP, CP):
        print("GPU Critical Pair: ")
        for i, part in enumerate(gcp):
            print(i, part)
        print('-------------------')
        print("Original Critical Pair {}:".format(i))
        for i, part in enumerate(cp):
            print(i, part)
        print('-------------------')
