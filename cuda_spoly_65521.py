#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from itertools import chain

from sympy import *
from sympy.polys.groebnertools import *
from sympy.polys.orderings import monomial_key

import numpy as np
from numba import cuda


def cuda_s_poly(cp, r):
    """
    Execute as a script to test.
    Called slightly differently from s_poly,
    must include ring.

    Prepare the data for the s-polynomial

    Create numpy arrays to send to gpu
    f, g, and dest arrays must all be the same length,

    figuring out the exact required output dimensions
    of the spoly procedure is exactly the F4 symbolic
    preprocessing step, and I don't know of another
    way to do it. Only the subtraction step of spoly
    is carried out on the GPU because of this, but
    it provides a micro demonstration of an F4 style
    matrix reduction. 


    Input: cp : a critical pair
           ring: for ordering, modulus
    """
    # Left and right of critical pair
    Ld = [(cp[0], cp[1], cp[2]), (cp[0], cp[4], cp[5])]

    # Get Length/monomials of destination array
    spair_info = symbolic_preprocessing(Ld, B, r)

    gpu_spoly = spoly_numba_io(spair_info, r)

    return gpu_spoly


def cuda_s_poly2(cp, r):
    """
    Another version of s_poly that
    just calculates each step in separate kernels.
    and reindexes the monomials on the host
    in between. May be improved by use of
    a cuda stream in CUDA-C or PyCUDA
    """
    order = r.order
    modulus = r.domain.mod
    nvars = len(r.symbols)

    # Multiply step
    f = cp[2]
    g = cp[5]

    fsig_idx = Sign(f)[1]
    gsig_idx = Sign(g)[1]
    fnum = Num(f)
    gnum = Num(g)

    # Fix this later
    # Here's the PROBLEM
    # Coefficient Values may come in negative, and will first be modded
    # Modulo INT_MAX_LENGTH. 14 is congruent to -1 mod INTMAX mod 65521
    # use a larger int that supports negative numbers for coeffs?
    # Coefficient arrays should be int32 to catch negative numbers and
    # integer overflow
    um = list(cp[1][0])
    vm = list(cp[4][0]) # separate 
    uc = cp[1][1]
    vc = cp[4][1]
    um = np.array(um, dtype=np.uint16)
    vm = np.array(vm, dtype=np.uint16)

    uv_coeffs = [uc, vc]
    uv_coeffs = np.array(uv_coeffs, dtype=np.int32)

    fsm = np.array([Sign(f)[0]] + Polyn(f).monoms(), dtype=np.uint16)
    gsm = np.array([Sign(g)[0]] + Polyn(g).monoms(), dtype=np.uint16)

    fc = [f for f in Polyn(f).coeffs()]
    fc = np.array(fc, dtype=np.int32)
    gc = [g for g in Polyn(g).coeffs()]
    gc = np.array(gc, dtype=np.int32)

    fsm_dest = np.zeros_like(fsm)
    gsm_dest = np.zeros_like(gsm)
    fc_dest = np.zeros_like(fc)
    gc_dest = np.zeros_like(gc)

    # launch kernel
    spoly_mul_numba_kernel(fsm_dest, gsm_dest, fc_dest, gc_dest,
                           fsm, gsm, fc, gc, um, vm, uv_coeffs, nvars, modulus)

    # Sub Step
    # Get all monomials in both umf, vmg, sort by ordering, reindex
    # f, g in a 2d coefficient array, send to other kernel
    
    fnew = [tuple(f) for f in fsm_dest]
    gnew = [tuple(g) for g in gsm_dest]
    fnew_sig = fnew[0]
    gnew_sig = gnew[0]
    fnew_monoms = [f for f in fnew[1:]]
    gnew_monoms = [g for g in gnew[1:]]

    all_monoms = set(fnew_monoms).union(set(gnew_monoms))
    all_monoms = sorted(all_monoms, key=monomial_key(order=r.order), reverse=True)

    spair_matrix = np.zeros((2, len(all_monoms)), dtype=np.int32)

    for fm, fc in zip(fnew_monoms, fc_dest):
        spair_matrix[0, all_monoms.index(fm)] = fc
    for gm, gc in zip(gnew_monoms, gc_dest):
        spair_matrix[1, all_monoms.index(gm)] = gc

    # Parse
    spair_info = parse_gpu_spoly_mul(spair_matrix, all_monoms, fnew_sig,
                                     gnew_sig, fsig_idx, gsig_idx, fnum,
                                     gnum, r)

    lb_spoly = spoly_numba_io(spair_info, r)

    return lb_spoly

def spoly_numba_io(spair_info, r):
    """
    Prepare the mini macaulay matrix for the numba kernel
    Called after symbolic_preprocessing only.

    Coefficient values should be of size int32
    """
    modulus = r.domain.mod

    cols = spair_info["cols"]
    rows = spair_info["rows"]

    spair_matrix = np.zeros((rows, cols), dtype=np.int32)
    dest = np.zeros(cols, dtype=np.int32)

    # fill at coordinates with nonzero entries
    for coords in spair_info["nze"]:
        spair_matrix[coords[0][0], coords[0][1]] = coords[1]

    spoly_sub_numba_kernel(dest, spair_matrix, modulus)

    # parse
    lb_spoly = parse_gpu_spoly(dest, spair_info, r)

    return lb_spoly


@cuda.jit
def spoly_sub_numba_kernel(dest, spair, modulus):
    """
    Basically Micro F4 partial reduction

    Subtracts f from g and stores in dest
    spair is a 2-row macaulay matrix of 
    coefficients in f and g in given monomial ordering.
    
    Likely grossly inefficient compared to CPU due
    to memory access times, but parallel. Demonstrates
    part of the process of F4 reduction.
    """
    for i in range(dest.size):
        dest[i] = ((spair[0][i] % modulus) - (spair[1][i] % modulus)) % modulus


@cuda.jit
def spoly_mul_numba_kernel(fsm_dest, gsm_dest, fc_dest, gc_dest,
                           fsm, gsm, fc, gc, um, vm, uv_coeffs, nvars, modulus):
    """
    Numba lbp_mul kernel for cuda_s_poly2. 
    Stage one of Spoly, 
    fsm_dest, gsm_dest must be made a set, sorted, 
    and fc, gc reindexed into a 2d array for 
    sub step kernel

    Bug: Elimination of LM. Should happen by construction
         Likely from incorrect inversion operation
         Keeping the constant term. Still don't know
    """
    # multiply um by fsm, vm by gsm
    frows = fsm.shape[0]
    for j in range(frows):
        for i in range(nvars):
            fsm_dest[j, i] = ((um[i] % modulus) + (fsm[j, i] % modulus)) % modulus

    grows = gsm.shape[0]
    for j in range(grows):
        for i in range(nvars):
            gsm_dest[j, i] = ((vm[i] % modulus) + (gsm[j, i] % modulus)) % modulus

    # multiply coefficients
    for i in range(1, fc_dest.size):
        fc_dest[i] = ((uv_coeffs[0] % modulus) * (fc[i] % modulus)) % modulus

    for i in range(1, gc_dest.size):
        gc_dest[i] = ((uv_coeffs[1] % modulus) * (gc[i] % modulus)) % modulus


def symbolic_preprocessing(Ld, B, r):
    """
    Mini Symbolic Preprocessing for Single S-Polynomial
    
    Input: Ld     : two 3-tuples(sig, um, f), (sig, vm, g)
           B      : intermediate basis
           ring   : for domain, order stuff
    
    Out: Information needed to construct a macaulay matrix.
    """
    order = r.order
    domain = r.domain

    Fi = set([lbp_mul_term(sc[2], sc[1]) for sc in Ld])
    Done = set([Polyn(f).LM for f in Fi])
    M = [Polyn(f).monoms() for f in Fi]
    M = set([i for i in chain(*M)]).difference(Done)
    while M != Done:
        MF = M.difference(Done)
        if MF != set():
            m = MF.pop()
            Done.add(m)
            for g in B:
                if monomial_divides(Polyn(g).LM, m):
                    u = term_div((m, domain.one), Polyn(g).LT, domain)
                    ug = (lbp_mul_term(g, u))
                    for m in Polyn(ug).monoms():
                        M.add(m)
        else:
            break

    # Fi sorted by sig_key, normalized, labeled, Done by monomial order
    Fi = sorted(Fi, key=lambda f: sig_key(f[0], r.order), reverse=True)
    Fi = [lbp(Sign(f), Polyn(f).monic(), Num(f)) for f in Fi]
    Done = sorted(Done, key=monomial_key(order=r.order), reverse = True)

    # pseudo COO sparse format
    nonzero_entries = []
    for i, f in enumerate(Fi):
        for t in Polyn(f).terms():
            nonzero_entries.append(((i, Done.index(t[0])), t[1]))

    spair_info = dict()
    spair_info["cols"] = len(Done)
    spair_info["rows"] = len(Fi)
    spair_info["nze"] = nonzero_entries
    spair_info["monomials"] = Done
    spair_info["spair"] = Fi

    return spair_info


def parse_gpu_spoly(dest, spair_info, r):
    """
    Return GPU spoly to sympy labeled polynomial

    Input: dest : the destination array from kernel
           spair_info: from symbolic_preprocessing
           ring : ordering, domain, etc.

    Output: sympy lbp 3 tuple (sig, poly, num)
    """
    spoly_sig = spair_info["spair"][0][0]
    spoly_num = spair_info["spair"][0][2]

    pexp = []
    for i, c in enumerate(dest):
        if c != 0:
            pexp.append('+' + str(c))
            for j, e in enumerate(spair_info["monomials"][i]):
                if e != 0:
                    pexp.append('*' + str(r.symbols[j]) + '**' + str(e))
    spol = r.from_expr(''.join(pexp))
    lb_spol = tuple([spoly_sig, spol, spoly_num])
    return lb_spol


def parse_gpu_spoly_mul(spair_matrix, all_monoms, fnew_sig, gnew_sig,
                        fsig_idx, gsig_idx, fnum, gnum, r):
    """
    parse into same output as symbolic_preprocessing to reuse
    numba_spoly_io function. Contains some redundant information,
    optimize later.
    """
    fsig = (tuple(fnew_sig), fsig_idx)
    gsig = (tuple(gnew_sig), gsig_idx)

    fpexp = []
    gpexp = []

    for i, c in enumerate(spair_matrix[0]):
        if c != 0:
            fpexp.append('+' + str(c))
            for j, e in enumerate(all_monoms[i]):
                if e != 0:
                    fpexp.append('*' + str(r.symbols[j]) + '**' + str(e))

    fp = r.from_expr(''.join(fpexp))
    lbf = (fsig, fp, fnum)

    for i, c in enumerate(spair_matrix[1]):
        if c != 0:
            gpexp.append('+' + str(c))
            for j, e in enumerate(all_monoms[i]):
                if e != 0:
                    gpexp.append('*' + str(r.symbols[j]) + '**' + str(e))

    gp = r.from_expr(''.join(gpexp))
    lbg = (gsig, gp, gnum)

    spair = [lbf, lbg]
    spair = sorted(spair, key=lambda f: sig_key(f[0], r.order), reverse=True)

    nze = []
    for i, f in enumerate(spair):
        for t in Polyn(f).terms():
            nze.append(((i, all_monoms.index(t[0])), t[1]))

    spair_info = dict()
    spair_info["rows"] = 2
    spair_info["cols"] = len(all_monoms)
    spair_info["spair"] = spair
    spair_info["monomials"] = all_monoms
    spair_info["nze"] = nze

    return spair_info


if __name__ == "__main__":
    print("CUDA Spoly Test")

    test_r, a, b, c, d, e = ring(symbols='a, b, c, d, e', domain=GF(65521), order='grevlex')
    """
    print("Cyclic Affine 4")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    """

    print("Cyclic Homogeneous 4")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*c + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - e**4
    F = [f1, f2, f3, f4]

    order = test_r.order

    B = [lbp(sig(test_r.zero_monom, i), f, i) for i, f in enumerate(F)]
    B = sorted(B, key=lambda g: order(Polyn(g).LM), reverse=True)

    CP = [critical_pair(B[i], B[j], test_r)
          for i in range(len(B)) for j in range(i + 1, len(B))]
    CP = sorted(CP, key=lambda cp: cp_key(cp, test_r), reverse=True)

    S = [cuda_s_poly(CP[i], test_r) for i in range(len(CP))]
    S_orig = [s_poly(CP[i]) for i in range(len(CP))]

    print("Output of original s_poly")
    for i, orig_s in enumerate(S_orig):
        print(i)
        for j, c in enumerate(orig_s):
            print(j, c)

    print("Output of cuda_s_poly")
    for i, s in enumerate(S):
        print(i)
        for j, c in enumerate(s):
            print(j, c)

    assert(set(S) == set(S_orig))
