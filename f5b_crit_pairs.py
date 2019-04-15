#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sympy import *
from sympy.polys.groebnertools import *
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from numba import cuda, jit, vectorize, guvectorize


def npairs(n):
    return abs(n*(n-1)//2)


def pairlist(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def modified_pair(pair):
    """
    Returns truncated lbp pair for transformation
    to cuda arrays.
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
    cuda_cp_array = np.zeros(2 * nvars + 1, dtype=np.float32)
    
    # Signature Multiplier
    for i, s in enumerate(mod_pair_part[0]):
        cuda_cp_array[i] = s

    # Leading Monomial - Remember stride of nvars
    for i, e in enumerate(mod_pair_part[1][0]):
        cuda_cp_array[i + nvars] = e

    # Leading Coefficient
    cuda_cp_array[-1] = mod_pair_part[1][1]

    return cuda_cp_array


def cp_cuda_prepare(mod_pair, ring):
    """
    First version of critical_pair for CUDA.
    Prepares data to send to PyCUDA kernel.
    1D forms core functionality, 2D are data
    independent.
    """
    nvars = len(ring.symbols)  # effectively our stride
    lt_buf = np.zeros(nvars + 1, dtype=np.float32)  # for lt in orig
    lt_buf[-1] = 1  # lt's coeff is defined as ring.domain.one
    um_buf = np.zeros_like(lt_buf)
    vm_buf = np.zeros_like(lt_buf)
    fdest = np.zeros(2 * nvars + 1, dtype=np.float32)  # storing results
    gdest = np.zeros_like(fdest)
    # Fill the F, G
    f = get_cuda_cp_array(mod_pair[0], nvars)
    g = get_cuda_cp_array(mod_pair[1], nvars)

    print("--------INPUT--------")
    print("nvars: ", nvars)
    print("f: ", f)
    print("g: ", g)
    print("lt_buf: ", lt_buf)
    print("um_buf: ", um_buf)
    print("vm_buf: ", vm_buf)
    print("fdest: ", fdest)
    print("gdest: ", gdest)
    print("---------------------")
    
    kernel_data = [nvars, lt_buf, um_buf, vm_buf, f, g, fdest, gdest]

    print("--------OUTPUT-------")
    # numba kernels can't have a return value
    # probably modifies input data in place
    finished_numba_dest_arrs = numba_cp_kernel_launch(kernel_data)
    print("Numba Finished Dest Arrays")
    for i, p in enumerate(finished_numba_dest_arrs):
        print("f{}: ".format(i), p)
    print("---------------------")
    # finished_pycuda_dest_arr = pycuda_cp_kernel(kernel_data)
    # print("PyCUDA finished: ", finished_pycuda_dest_arr)
    """
    TODO: Parse data back out of dest array.
    """
    return


def numba_cp_kernel_launch(kernel_data):
    """
    Prepared nparray data for numba cuda jit
    Appears to modify sent arrays in place from
    their documentation
    """
    nvars = kernel_data[0]
    lt_buf = kernel_data[1]
    um_buf = kernel_data[2]
    vm_buf = kernel_data[3]
    f = kernel_data[4]
    g = kernel_data[5]
    fdest = kernel_data[6]
    gdest = kernel_data[7]

    numba_critical_pairs(nvars, lt_buf, um_buf, vm_buf,
                         f, g, fdest, gdest)
    return [fdest, gdest]


@cuda.jit
def numba_critical_pairs(nvars, lt_buf, um_buf, vm_buf,
                         f, g, fdest, gdest):
    """
    Numba Cuda.Jit kernel for critical pair computation.

    INPUT:
    nvars: integer, used as array stride

    lt_buf: intermediate storage for monomial_lcm(f, g) (len nvars + 1)
               lt_buf[:nvars] : monomial
               lt_buf[-1] : 1 (ring.domain.one)
        * This should be in the shared memory for all threads
          after computation

    um_buf: intermediate storage for um/vm init to zeros

    f, g : polynomials for cp computation of len 2*nvars + 1
           f[0:nvars] : signature multiplier field
           f[nvars:-1] : leading monomial of f
           f[-1] : leading coefficient of f

    fdest : a destination array for final result
           dest[:nvars] : sig(fr) multiplier field
           dest[nvars:2*nvars+1] : um field
           dest[nvars:2*nvars] : um monomial field
           dest[2*nvars] : um coefficient
           similar for g

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
    # Going to start by just writing out sequentially and seeing if numba
    # will vectorize or auto-jit it for us.
    for i in range(nvars):
        lt_buf[i] = max(f[nvars + i], g[nvars + i])

    cuda.syncthreads()

    for i in range(nvars):
        um_buf[i] = lt_buf[i] - f[nvars + i]
        vm_buf[i] = lt_buf[i] - g[nvars + i]

    um_buf[-1] = lt_buf[-1] / f[-1]
    vm_buf[-1] = lt_buf[-1] / g[-1]

    cuda.syncthreads()

    for i in range(nvars):
        fdest[i] = f[i] + um_buf[i]
        gdest[i] = g[i] + vm_buf[i]

    fdest[-1] = um_buf[-1]
    gdest[-1] = vm_buf[-1]


def pycuda_cp_kernel(kernel_data):
    """
    PyCUDA Kernel for Critical Pair computation.
    Attempt to maximize thread use.

    Procedure:
    1: Send nvars, lt_buf, f, g, dest to device
    2: put lt_buf, dest into shared memory
    3: compute lt, store results in lt_buf
    4: Synchronize Threads
    5: Simultaneously compute um, vm, store in dest[nvars, 2*nvars+1],
       and dest[3*nvars+1, :] respectively.
    6: Synchronize Threads
    7: Simultaneously Compute sig(fr, gr), store in dest[0:nvars], 
       dest[2*nvars+1:3*nvars+1] respectively
    8: Synchronize threads
    9: Return to Host device
    10: on host, check for valid term division in um, vm. (nonnegative)
    11: Send back to other function to parse back into cp 6-tuple form.
    """
    mod = SourceModule("""
    __global__ void critical_pair(int nvars, float *lt_buf, float *f, float *g)
    {
    const int i threadIdx.x;
    

    }
    """)
    cuda_critical_pair = mod.get_function("critical_pair")

    nvars = kernel_data[0]
    lt_buf = kernel_data[1]
    f = kernel_data[2]
    g = kernel_data[3]
    fdest = kernel_data[4]
    gdest = kernel_data[5]
    
    cuda_critical_pair(
        drv.Out(fdest), drv.Out(gdest),
        drv.In(nvars), drv.In(lt_buf), drv.In(f), drv.In(g),
        block=(400, 1, 1), grid=(1, 1)
    )
    return [fdest, gdest]


if __name__ == "__main__":
    r, a, b, c, d = ring(symbols="a, b, c, d", domain="RR", order="grevlex")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    F = [f1, f2, f3, f4]

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]
    for i in range(len(B)):
        print(i, B[i])
    pairs = [(B[i], B[j]) for (i, j) in pairlist(len(B))]
    assert(len(pairs) == npairs(len(B)))

    for p in pairs:
        cp_cuda_prepare(modified_pair(p), r)
