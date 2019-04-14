#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
from itertools import chain
from ordered_set import OrderedSet

from sympy import *
from sympy.polys.groebnertools import *
from sympy.polys.orderings import monomial_key
from sympy.polys.domains import QQ, RR
from sympy import init_printing, pprint

import numpy as np
from numba import cuda, jit, vectorize, guvectorize

##########################################################
#    GPU-based f5b
# Major Changes:
# *Vectorized/JIT compiled Critical Pairs
# *Vectorized/JIT compiled s-polynomial computation
# *F4/F5B style matrix reduction
# *F5C Interreduced Basis Optimization
#
# *Notes: Added optional args to pass to run() to select
#         computation mode. ('sympy', 'numpy', 'pycuda')
##########################################################

def _f5b_gpu(F, ring, cuda=False, mode=None, D=None, incremental=False):
    domain, orig = ring.domain, None

    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, ring = ring, ring.clone(domain=domain.get_field())
        except DomainError:
            raise DomainError("can't compute a Groebner basis over %s" % domain)
        else:
            F = [ s.set_ring(ring) for s in F ]

    order = ring.order

    # reduce polynomials (like in Mario Pernici's implementation) (Becker, Weispfenning, p. 203)
    B = F
    while True:
        F = B
        B = []

        for i in range(len(F)):
            p = F[i]
            r = p.rem(F[:i])

            if r:
                B.append(r)

        if F == B:
            break

    # basis
    B = [lbp(sig(ring.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)

    # critical pairs (cuda_cp removed for now)
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

    k = len(B)

    reductions_to_zero = 0

    # Branch to CUDA F5B
    if cuda == True and domain.is_Field:
        d = D
        cuda_D_GB = cuda_f5b(CP, B, k, reductions_to_zero, ring, mode=mode, D=d)
    elif cuda == True and not domain.is_Field:
        print("CUDA F5B only supports fields")
        return None

    if incremental == False:
        return None
    
    
    while len(CP):
        cp = CP.pop()

        # discard redundant critical pairs:
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue
        
        s = s_poly(cp)

        p = f5_reduce(s, B)

        p = lbp(Sign(p), Polyn(p).monic(), k + 1)

        if Polyn(p):
            # remove old critical pairs, that become redundant when adding p:
            indices = []
            for i, cp in enumerate(CP):
                if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                    indices.append(i)
                elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                    indices.append(i)

            for i in reversed(indices):
                del CP[i]

            # only add new critical pairs that are not made redundant by p:
            for g in B:
                if Polyn(g):
                    cp = critical_pair(p, g, ring)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue

                    CP.append(cp)

            # sort (other sorting methods/selection strategies were not as successful)
            CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

            # insert p into B:
            m = Polyn(p).LM
            if order(m) <= order(Polyn(B[-1]).LM):
                B.append(p)
                # Interreduce Basis
                B = interreduce_basis(B, ring)
                k = len(B)
                continue
            else:
                for i, q in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break
                B = interreduce_basis(B, ring)
                k = len(B)
                continue
            k += 1

            #print(len(B), len(CP), "%d critical pairs removed" % len(indices))
        else:
            reductions_to_zero += 1

    # reduce Groebner basis: Computing with Reduced GBs throughout
    # Reduces reductions to zero by a noticeable amount.
    print("---------------")
    print("Sympy F5B-Incremental Output")
    print("Interreduced Grobner Basis")
    for i, b in enumerate(B):
        print(i, b)
    print("Zero Reductions: " + str(reductions_to_zero))
    print("---------------")
    return sorted([Polyn(g).monic() for g in B], key=lambda f:order(f.LM), reverse=True)


def cuda_f5b(CP, B, k, reductions_to_zero, ring, mode=None, D=None):
    """
    Has an F4 like strategy for simultaneously reducing critical pairs.
    Maintains signatures through matrix reduction, can still use F5B criteria
    """
    order = ring.order
    #print("Beginning of CP loop")
    #for i, cp in enumerate(CP):
    #    print(i, cp)
    #stop = input("Press enter to continue")

    d = 0
    while CP:
        #print("CP While loop start")
        #print("Total CPs: {}".format(len(CP)))
        CP = OrderedSet(CP)  # Maintain our ordering from before.
                
        cp_remove = OrderedSet([cp for cp in CP if is_rewritable_or_comparable(cp[0], Num(cp[2]), B)])
        CP = CP.difference(cp_remove)
        cp_remove = OrderedSet([cp for cp in CP if is_rewritable_or_comparable(cp[3], Num(cp[5]), B)])
        CP = CP.difference(cp_remove)

        if D:
            if D <= d:
                print("--------------------")
                print("Degree Bounded {}-GB".format(d))
                H = interreduce_basis(B, ring)
                for i, g in enumerate(H):
                    print(i, g)
                print("Zero Reductions: {}".format(reductions_to_zero))
                return H
            else:
                print("Degree reached: {}".format(d))

        d = minimal_degree(CP, ring)
        CPd = Sel(CP, d, ring)
        CP = CP.difference(CPd)
        
        #print("CPd: Selection of critical pairs of minimal degree")
        #for i, cp in enumerate(CPd):
        #    print(i, cp)
        #stop = input("press enter to continue")
        
        # Take the set of CP components for creating Macaulay Matrix
        CPleft = OrderedSet([(cp[0], cp[1], cp[2]) for cp in CPd])
        CPright = OrderedSet([(cp[3], cp[4], cp[5]) for cp in CPd])
        Ld = CPleft.union(CPright) # maybe just pass this

        # reduction stage
        # Performs symbolic preprocessing to create the coefficient matrix
        # Then performs Gaussian Elimination on the matrix and returns the new polynomials
        Ftilde = cuda_reduction(Ld, B, ring, mode)

        Ftilde_polys = [f for f in Ftilde if Polyn(f)]
        cp_remove = set([cp for cp in CP
                         if is_rewritable_or_comparable(cp[0], Num(cp[2]), Ftilde_polys)])
        cp_remove = cp_remove.union(set([cp for cp in CP
                                         if is_rewritable_or_comparable(cp[3], Num(cp[5]),
                                                                        Ftilde_polys)]))
        CP = CP.difference(cp_remove)
        CP = sorted(CP, key=lambda cp: cp_key(cp, ring), reverse=True)

        for i, h in enumerate(Ftilde):
            if Polyn(h):
                h = lbp(Sign(h), Polyn(h).monic(), k)

                for g in B:
                    if Polyn(g):
                        cp = critical_pair(h, g, ring)
                        if is_rewritable_or_comparable(cp[0], Num(cp[2]), [h]):
                            continue
                        elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [h]):
                            continue
                        CP.append(cp)

                CP = sorted(CP, key=lambda cp: cp_key(cp, ring), reverse=True)

                m = Polyn(h).LM
                if order(m) <= order(Polyn(B[-1]).LM):
                    B.append(h)
                    B = interreduce_basis(B, ring)
                    k = len(B)
                else:
                    for i, q in enumerate(B):
                        if order(m) > order(Polyn(q).LM):
                            B.insert(i, h)
                            B = interreduce_basis(B, ring)
                            k = len(B)
                            break
                k += 1
            else:
                reductions_to_zero += 1

    H = sorted([g for g in B],
               key=lambda f: order(Polyn(f).LM), reverse=True)
    print("-----------------------")
    print("F5BC_GPU has terminated without bound D")
    print("Interreduced Grobner Basis")
    for i, h in enumerate(H):
        print(i, h)
    print("Zero Reductions: ", reductions_to_zero)
    print("Degree Reached: ", d)
    print("-----------------------")
    return H


# NEW FUNCS
def interreduce_basis(B, ring):
    """
    Reduces the incremental basis when adding
    new polynomials to it. Idea from F5C.
    Resets signatures, k to len(B), and reinitializes
    B as a reduced grobner basis within the loop
    rather than a final step.
    Might not work (they say you can dispose of the signatures,
    but they don't use the Num() field in their labeled polynomials
    The comparable idea of the list of rules is reset and however,
    so it may be fine, assuming the basis B grows in degree until
    it is a GB. After testing, it does appear to work well.
    """
    order = ring.order

    H = [Polyn(g).monic() for g in B]
    H = red_groebner(H, ring)
    B = [lbp(sig(ring.zero_monom, i + 1), H[i], i + 1)
         for i in range(len(H))]

    return sorted(B, key=lambda f: order(Polyn(f).LM), reverse=True)


def Sel(CP, d, ring):
    domain = ring.domain
    return OrderedSet([cp for cp in CP
                if max(monomial_lcm(Polyn(cp[2]).LM, Polyn(cp[5]).LM)) == d])


def minimal_degree(CP, ring):
    if len(CP) == 0:
        return 0
    domain = ring.domain
    D = []
    for cp in CP:
        ltf = Polyn(cp[2]).LT
        ltg = Polyn(cp[5]).LT
        lt = monomial_lcm(ltf[0], ltg[0])
        D.append(lt)
    d = min(D)
    d = max(d)
    return d


def cuda_reduction(Ld, B, ring, mode=None):
    """
    In: set of S-poly components, intermediate basis B
    """

    matrix_info = symbolic_preprocessing(Ld, B, ring)
    Ft = cuda_gauss_elimination(matrix_info, ring, mode)
    """
    print("----------OUTPUT SYSTEM-----------")
    for i, f in enumerate(Ft):
        print(i, f[0], f[1], f[2])
    print("----------------------------------")
    stop = input("press enter to continue")
    """
    return Ft


def symbolic_preprocessing(Ld, B, ring):
    """
    Need to prepare what we need for constructing our matrix
    All monomials in the system, all available reductors
    """
    domain = ring.domain

    Fi = OrderedSet([lbp_mul_term(tf[2], tf[1]) for tf in Ld])
    Done = OrderedSet([Polyn(f).LM for f in Fi])
    M = [Polyn(f).monoms() for f in Fi]
    M = OrderedSet([i for i in chain(*M)]).difference(Done)
    while M != Done:
        MF = M.difference(Done)
        if MF != set():
            m = MF.pop()
            Done.add(m)
            for g in B:
                if monomial_divides(Polyn(g).LM, m):
                    u = term_div((m, domain.one), Polyn(g).LT, domain)
                    ug = (lbp_mul_term(g, u))
                    Fi.add(ug)
                    for m in Polyn(ug).monoms():
                        M.add(m)
        else:
            break

    # Create macaulay matrix from information derived here
    # Collect matrix shape, preprepare matrix by sorting rows by
    # weight (number of nonzero entries), heaviest first.
    # Make all polynomials monic, so that all pivoting candidates are 1
    # Store (indices, coeff) of all nonzero entries
    Matrix_Info = dict()
    Fi = sorted(Fi, key=lambda f: len(Polyn(f).monoms()), reverse=True)
    Fi = [lbp(Sign(f), Polyn(f).monic(), Num(f)) for f in Fi]
    Done = sorted(Done, key=monomial_key(order=ring.order), reverse=True)

    """
    print("----------------------------")
    print("System Fi:")
    print("----------------------------")
    for i, f in enumerate(Fi):
        print(i, f)
    print("----------------------------")
    print("Monomials in Done")
    print("----------------------------")
    for i, m in enumerate(Done):
        print(i, m)
    print("----------------------------")
    stop = input("Press enter to continue")
    """
    pivoting_candidates = []
    nonzero_entries = []
    for i, f in enumerate(Fi):
        pivoting_candidates.append((i, Done.index(Polyn(f).LM)))
        for t in Polyn(f).terms():
            nonzero_entries.append(((i, Done.index(t[0])), t[1]))

    cols_last_nze = []
    for col in range(len(Done)):
        max_row = 0
        for nze in nonzero_entries:
            if max_row < nze[0][0]:
                max_row = nze[0][0]
        cols_last_nze.append((max_row, col))
    permutable_cols = [p[1] for p in cols_last_nze
                       if p not in pivoting_candidates]

    Matrix_Info["shape"] = (len(Fi), len(Done))
    Matrix_Info["rows"] = len(Fi)
    Matrix_Info["cols"] = len(Done)
    Matrix_Info["total_entries"] = len(Fi) * len(Done)
    Matrix_Info["pivoting_candidates"] = pivoting_candidates
    Matrix_Info["nonzero_entries"] = nonzero_entries
    Matrix_Info["num_nze"] = len(nonzero_entries)
    Matrix_Info["permutable_cols"] = permutable_cols
    Matrix_Info["monomials"] = Done
    Matrix_Info["input_system"] = Fi

    """
    print("----------------------")
    print("Coefficient Matrix Info")
    for k, v in Matrix_Info.items():
        print(str(k) + ': ' + str(v))
    """
    return Matrix_Info


def cuda_gauss_elimination(matrix_info, ring, mode=None):
    """
    Matrix Info as a naive representation of GBLA
    data format to avoid constructing giant zero matrices
    as much as possible.
    
    Start with NP.zeros array, start with float,
    try to adapt to ring.domain type if possible.

    Need to figure out basic operations over Galois Fields,
    Stick with QQ, RR for now.

    Note: Most recent CUDA platform with RTX cards has
    optimized performance with half precision floats.
    May be something to test.
    """
    domain = ring.domain
    if domain not in [QQ, RR]:
        print("Use QQ or RR for domain for now")
        return None

    if mode not in ['sympy', 'numpy', 'pycuda']:
        mode = 'sympy'

    # Just for fun, and getting results back from matrix form.
    # SymPys matrix objects do not properly check for zeros, so
    # They're not going to be reliable.
    if mode == 'sympy':
        sp_A = zeros(matrix_info["rows"], matrix_info["cols"])
        for nze in matrix_info["nonzero_entries"]:
            sp_A[nze[0][0], nze[0][1]] = nze[1]

        # Since we're only working with QQ, RR, and float
        # sensible fields RREF should always be fine here.
        # RREF_OUT is a tuple of rref matrix and the pivots.
        rref_out = sp_A.rref()

        """
        print("----------------")
        print("Sympy Mode")
        print("----------------")
        print("Sympy Macaulay Matrix")
        pprint(sp_A)
        print("----------------")
        print("RREF")
        pprint(rref_out[0])
        print("Pivots")
        pprint(rref_out[1])
        if (rref_out[0].rows - len(rref_out[1])):
            print("-------------")
            print("{} Zero Reduction".format(rref_out[0].rows - len(rref_out[1])))
        print("----------------")
        stop = input("press enter to continue")
        """
        return symbolic_postprocessing(matrix_info,
                                       rref_out[0], ring)

    if mode == 'numpy':
        # First up: Naive Gauss with np.ndarray
        np_A = np.zeros(matrix_info["shape"], dtype=np.float32)
        np_dest = np_A.copy()

        # Fill up the matrix with the correct nonzero entries
        # A nonzero entry is ((row, col), coeff)
        for nze in matrix_info["nonzero_entries"]:
            np_A[nze[0][0], nze[0][1]] = np.float32(nze[1])
        print('-------------------')
        print('Coefficient Matrix')
        print('-------------------')
        print(np_A)

        reduced_A = np_naive_gaussian_elimination(np_dest, np_A)

        print("This is as far as we've gotten for this.")
        sys.exit(0)
        return None

    if mode == 'pycuda':
        # Hopefully we get there
        print("Not yet implemented")
        sys.exit(0)
        return None

    # Until we have something to return
    return None


def symbolic_postprocessing(matrix_info, reduced_matrix, ring):
    """
    Rebuilds PolyElement objects from reduced matrix
    Works for sympy matrix so far

    Zero reductions separated from Ftilde for sorting purposes.
    Zero reductions is still in lbp form with the sig and num
    of the corresponding zero-reduced polynomial as None.
    """
    Ftilde = []
    Zero_Reductions = []
    for i in range(reduced_matrix.rows):
        if all([a == 0 for a in reduced_matrix.row(i)]):
            Zero_Reductions.append((matrix_info["input_system"][0],
                                    None, matrix_info["input_system"][i][2]))
            continue
        pexp = []
        for j, entry in enumerate(reduced_matrix.row(i)):
            pexp.append('+' + str(Rational(entry).limit_denominator(10*6)))
            for k, e in enumerate(matrix_info["monomials"][j]):
                pexp.append('*' + str(ring.symbols[k]) + '**' + str(e))
        if pexp != []: # ? filters zero rows??/
            pexp = ring.from_expr(''.join(pexp))
            labeled_pexp = lbp(matrix_info["input_system"][i][0],
                               pexp, matrix_info["input_system"][i][2])
            Ftilde.append(labeled_pexp)
        else:
            Zero_Reductions.append((matrix_info["input_system"][i][0],
                                    None, matrix_into["input_system"][i][2]))
    # Return sorted list, including zero reductions.
    Ftilde = sorted(Ftilde, key=lambda f: Polyn(f).LM, reverse=True)
    for z in Zero_Reductions:
        Ftilde.append(z)
    return Ftilde


###
# Numpy/Numba jit functions
###
@jit(nopython=True, fastmath=True, parallel=False)
def np_naive_gaussian_elimination(dest, A):
    """
    Notes:
     * numba cpu jit with Parallel set to True causes
       division by zero errors. Compiler may not be smart
       enough to parallelize the correct inner loops.

    Simple Gaussian Elimination
    Adapted from Alg 1.1: Naive Gaussian Elimination
    in Faugere-Lacharte Parallel Gaussian Elimination
    for Grobner Bases Computations over Finite Fields
    by Martani Fayssal

    Pseudocode:
    -----------
    r <-- 0
    for i = 1 to m do:
        piv_found <-- false
        for j = r + 1 to n do:
            if A[j, i] != 0 then:
                r <-- r + 1
                A[j, :] <---> A[r, :]
                A[r, :] <-- A[r, i]^-1 * A[r, :]
                piv_found <-- True
        if piv_found = True:
            for j = r + 1 to n do:
                A[j, :] <-- A[j, :] - A[j, i] * A[r, :]
    """
    # Some operations available in numpy are not well liked by
    # the numba jit. Will try to keep it as simple as possible.
    n, m = A.shape
    r = 0
    for i in range(m):
        piv_found = False
        for j in range(r + 1, n):
            if A[j, i] != 0:
                r += 1
                temp = A[j, :].copy()
                A[j, :] = A[r, :]
                A[r, :] = temp
                A[r, :] = A[r, i]**-1 * A[r, :]
                piv_found = True
        if piv_found:
            for j in range(r + 1, n):
                A[j, :] = A[j, :] - A[j, i] * A[r, :]

    dest = A  # This has something to do with CUDA
    return dest


def np_gauss_jordan(dest, A):
    return dest


def np_gbla_style(matrix_info):
    return


###
# PyCUDA functions: Probably not
###
def pc_naive_gaussian_elimination(dest, A):
    return dest


def pc_gauss_jordan(dest, A):
    return dest


def pc_gbla_style(matrix_info):
    dest = [[]]
    return dest


def run(I, R, cuda=False, mode=None, D=None, incremental=False):
    return _f5b_gpu(I, R, cuda, mode, D, incremental=incremental)


if __name__ == "__main__":
    init_printing(use_latex=True)
    print("F5BC_GPU Test")
    print("-------------------------------------------")

    test_order = input("Choose an ordering (1: 'lex', 2: 'grlex', 3: 'grevlex'): ")
    if test_order == '':
        test_order = 'lex'
    elif test_order == '1':
        test_order = 'lex'
    elif test_order == '2':
        test_order = 'grlex'
    elif test_order == '3':
        test_order = 'grevlex'
    if test_order not in ['lex', 'grlex', 'grevlex']:
        print("Not a supported ordering.")
        sys.exit(0)

    test_domain = input("Choose a domain (1: 'QQ', 2: 'RR' (default GF(65521))): ")
    if test_domain == '':
        test_domain = GF(65521)
    elif test_domain == '1':
        test_domain = 'QQ'
    elif test_domain == '2':
        test_domain = 'RR'
    if test_domain not in ['QQ', 'RR', GF(65521)]:
        print("Not a supported domain.")
        sys.exit(0)

    test_mode = input("Choose a mode (1: 'sympy', 2: 'numpy', 3: 'pycuda', default: None): ")
    if test_mode == '':
        test_mode = None
    elif test_mode == '1':
        test_mode = 'sympy'
    elif test_mode == '2':
        test_mode = 'numpy'
    elif test_mode == '3':
        test_mode = 'pycuda'
    if test_mode not in [None, 'sympy', 'numpy', 'pycuda']:
        print("Not a supported mode.")
        sys.exit(0)

    test_bound = input("Enter a degree bound (Default: None): ")
    if test_bound == '':
        test_bound = None
    elif int(test_bound) < 1:
        test_bound = None
    else:
        test_bound = int(test_bound)

    test_f5b = input("Include incremental F5B? (0: False, 1: True, Default: False): ")
    if test_f5b == '':
        test_f5b = False
    elif test_f5b == '0':
        test_f5b = False
    elif test_f5b == '1':
        test_f5b = True
    else:
        test_f5b = False

    #Cyclic-Affine-3
    CA3_R, x1, x2, x3 = ring(symbols="x1, x2, x3", domain=test_domain, order=test_order)
    f1 = x1 + x2 + x3
    f2 = x1*x2 + x1*x3 + x2*x3
    f3 = x1*x2*x3 - 1
    CA3_I = [f1, f2, f3]

    #Cyclid-Homogeneous-3
    CH3_R, x1, x2, x3, x4 = ring(symbols="x1, x2, x3, x4", domain=test_domain, order=test_order)
    f1 = x1 + x2 + x3
    f2 = x1*x2 + x1*x3 + x2*x3
    f3 = x1*x2*x3 - x4**3
    CH3_I = [f1, f2, f3]
        
    # Cyclic-Affine-4
    CA4_R, x1, x2, x3, x4 = ring(symbols="x1, x2, x3, x4", domain=test_domain, order=test_order)
    f1 = x1 + x2 + x3 + x4
    f2 = x1*x2 + x2*x3 + x1*x4 + x3*x4
    f3 = x1*x2*x3 + x1*x2*x4 + x1*x3*x4 + x2*x3*x4
    f4 = x1*x2*x3*x4 - 1
    CA4_I = [f1, f2, f3, f4]

    # Cyclic-Homogeneous-4
    CH4_R, x1, x2, x3, x4, x5 = ring(symbols="x1, x2, x3, x4, x5", domain=test_domain, order=test_order)
    f1 = x1+x2+x3+x4
    f2 = x1*x2+x2*x3+x1*x4+x3*x4
    f3 = x1*x2*x3+x1*x2*x4+x1*x3*x4+x2*x3*x4
    f4 = x1*x2*x3*x4-x5**4
    CH4_I = [f1, f2, f3, f4]

    # Cyclic-Homogeneous-10
    CH10_R, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = ring(symbols="x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11", domain=test_domain, order=test_order)
    f1 = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10
    f2 = x1*x2+x2*x3+x3*x4+x4*x5+x5*x6+x6*x7+x7*x8+x8*x9+x1*x10+x9*x10
    f3 = x1*x2*x3+x2*x3*x4+x3*x4*x5+x4*x5*x6+x5*x6*x7+x6*x7*x8+x7*x8*x9+x1*x2*x10+x1*x9*x10+x8*x9*x10
    f4 = x1*x2*x3*x4+x2*x3*x4*x5+x3*x4*x5*x6+x4*x5*x6*x7+x5*x6*x7*x8+x6*x7*x8*x9+x1*x2*x3*x10+x1*x2*x9*x10+x1*x8*x9*x10+x7*x8*x9*x10
    f5 = x1*x2*x3*x4*x5+x2*x3*x4*x5*x6+x3*x4*x5*x6*x7+x4*x5*x6*x7*x8+x5*x6*x7*x8*x9+x1*x2*x3*x4*x10+x1*x2*x3*x9*x10+x1*x2*x8*x9*x10+x1*x7*x8*x9*x10+x6*x7*x8*x9*x10
    f6 = x1*x2*x3*x4*x5*x6+x2*x3*x4*x5*x6*x7+x3*x4*x5*x6*x7*x8+x4*x5*x6*x7*x8*x9+x1*x2*x3*x4*x5*x10+x1*x2*x3*x4*x9*x10+x1*x2*x3*x8*x9*x10+x1*x2*x7*x8*x9*x10+x1*x6*x7*x8*x9*x10+x5*x6*x7*x8*x9*x10
    f7 = x1*x2*x3*x4*x5*x6*x7+x2*x3*x4*x5*x6*x7*x8+x3*x4*x5*x6*x7*x8*x9+x1*x2*x3*x4*x5*x6*x10+x1*x2*x3*x4*x5*x9*x10+x1*x2*x3*x4*x8*x9*x10+x1*x2*x3*x7*x8*x9*x10+x1*x2*x6*x7*x8*x9*x10+x1*x5*x6*x7*x8*x9*x10+x4*x5*x6*x7*x8*x9*x10
    f8 = x1*x2*x3*x4*x5*x6*x7*x8+x2*x3*x4*x5*x6*x7*x8*x9+x1*x2*x3*x4*x5*x6*x7*x10+x1*x2*x3*x4*x5*x6*x9*x10+x1*x2*x3*x4*x5*x8*x9*x10+x1*x2*x3*x4*x7*x8*x9*x10+x1*x2*x3*x6*x7*x8*x9*x10+x1*x2*x5*x6*x7*x8*x9*x10+x1*x4*x5*x6*x7*x8*x9*x10+x3*x4*x5*x6*x7*x8*x9*x10
    f9 = x1*x2*x3*x4*x5*x6*x7*x8*x9+x1*x2*x3*x4*x5*x6*x7*x8*x10+x1*x2*x3*x4*x5*x6*x7*x9*x10+x1*x2*x3*x4*x5*x6*x8*x9*x10+x1*x2*x3*x4*x5*x7*x8*x9*x10+x1*x2*x3*x4*x6*x7*x8*x9*x10+x1*x2*x3*x5*x6*x7*x8*x9*x10+x1*x2*x4*x5*x6*x7*x8*x9*x10+x1*x3*x4*x5*x6*x7*x8*x9*x10+x2*x3*x4*x5*x6*x7*x8*x9*x10
    f10 = x1*x2*x3*x4*x5*x6*x7*x8*x9*x10-x11**10
    CH10_I = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

    ####
    # Katsura Homogeneous 7
    ###
    KH7_R, x1, x2, x3, x4, x5, x6, x7, x8 = ring(symbols="x1, x2, x3, x4, x5, x6, x7, x8",
                                                 domain=test_domain, order=test_order)
    f1 = x1+2*x2+2*x3+2*x4+2*x5+2*x6+2*x7-x8
    f2 = x1**2-x1*x8+2*x2**2+2*x3**2+2*x4**2+2*x5**2+2*x6**2+2*x7**2
    f3 = 2*x1*x2+2*x2*x3-x2*x8+2*x3*x4+2*x4*x5+2*x5*x6+2*x6*x7
    f4 = 2*x1*x3+x2**2+2*x2*x4+2*x3*x5-x3*x8+2*x4*x6+2*x5*x7
    f5 = 2*x1*x4+2*x2*x3+2*x2*x5+2*x3*x6+2*x4*x7-x4*x8
    f6 = 2*x1*x5+2*x2*x4+2*x2*x6+x3**2+2*x3*x7-x5*x8
    f7 = 2*x1*x6+2*x2*x5+2*x2*x7+2*x3*x4-x6*x8
    KH7_I = [f1, f2, f3, f4, f5, f6, f7]

    run(CH4_I, CH4_R, cuda=True, mode=test_mode, D=test_bound, incremental=test_f5b)
