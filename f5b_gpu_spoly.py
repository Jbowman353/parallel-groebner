#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from itertools import chain
from sympy import *
from sympy.polys.groebnertools import *
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
from numba import cuda, jit, vectorize, guvectorize


def cuda_spoly_prepare(cp, ring):
    """
    Prepare the data for the s-polynomial

    Create flat numpy arrays to send to gpu
    f, g, and dest arrays must all be the same length,
    that is, the length of the signature multiplier plus
    (nvars + 1) all the monoms in both polynomials
    
    f and g's exponents and coefficients must be entered
    indexed by the ordering in the monomials.

    f and g's monomials might be separated from the coeffs
    to increase parallelization, both have same indices
    
    Effectively a two line matrix reduction.

    Need f_coeffs, g_coeffs, (flat) f_monoms, g_monoms,
    with the signatures on the front
    """
    order = ring.order
     
    nvars = len(ring.symbols)
    all_monoms = sorted(set(f.monoms()).union(g.monoms()),
                        key=lambda m: order(m), reverse=True)
    arylen = nvars + (nvars + 1) * len(all_monoms)
    dest = np.zeros(arylen, dtype=np.float32)

    # f, g signatures
    f_sig = list(Sign(cp[2])[0])
    g_sig = list(Sign(cp[5])[0])

    # index all coefficients and monomials in f, g
    f_indices = []
    g_indices = []

    for i, m in enumerate(all_monoms):
        for t in f.terms():
            if t[0] == m:
                f_indices.append(i)
        for t in g.terms():
            if t[0] == m:
                g_indices.append(i)

    # collect coefficients
    f_coeffs = [0] * len(all_monoms)
    g_coeffs = [0] * len(all_monoms)

    for i, c in zip(f_indices, f.coeffs()):
        f_coeffs[i] = c

    for i, c in zip(g_indices, g.coeffs()):
        g_coeffs[i] = c

    # collect all monomials
    f_monoms = [tuple([0] * nvars)] * len(all_monoms)
    g_monoms = [tuple([0] * nvars)] * len(all_monoms)
    
    for fidx in f_indices:
        f_monoms[fidx] = all_monoms[fidx]

    for gidx in g_indices:
        g_monoms[gidx] = all_monoms[gidx]

    # flatten monom array, stride by nvars
    f_monoms_flat = [f for f in chain(*f_monoms)]
    g_monoms_flat = [g for g in chain(*g_monoms)]

    # append the signature to the front
    fsig_monom = f_sig + f_monoms_flat
    gsig_monom = g_sig + g_monoms_flat

    # flat arrays for um, vm with monom, um_ary[-1] = coeff
    um_ary = list(cp[1][0]) + [cp[1][1]]
    vm_ary = list(cp[4][0]) + [cp[4][1]]

    # Convert Spoly info to numpy vectors
    np_fsig_monom = np.array(fsig_monom, dtype=np.float32)
    np_gsig_monom = np.array(gsig_monom, dtype=np.float32)
    np_fcoeffs = np.array(f_coeffs, dtype=np.float32)
    np_gcoeffs = np.array(g_coeffs, dtype=np.float32)
    np_um_ary = np.array(um_ary, dtype=np.float32)
    np_vm_ary = np.array(vm_ary, dtype=np.float32)

    # buffers for intermediate operations may be needed
    # package spoly_info to ship to a numba kernel launching function
    spoly_info = [np_fsig_monom, np_gsig_monom, np_fcoeffs, np_gcoeffs,
                  np_um_ary, np_vm_ary, dest]

    # for now, back to it tomorrow
    return spoly_info
    

if __name__ == "__main__":
    r, a, b, c, d = ring(symbols='a, b, c, d', domain='RR', order='grevlex')
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    F = [f1, f2, f3, f4]

    domain = r.domain
    order = r.order

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]
    for i, b in enumerate(B):
        print(i, b)

    CP = [critical_pair(B[i], B[j], r)
          for i in range(len(B)) for j in range(i + 1, len(B))]
    CP = sorted(CP, key=lambda cp: cp_key(cp, r), reverse=True)

    for i, cp in enumerate(CP):
        print(i, cp)

    cuda_spoly_prepare(CP[0], r)

    sys.exit(0)
