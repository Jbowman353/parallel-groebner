#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from sympy import Rational
from sympy.polys.domains import QQ, RR
from sympy.polys.groebnertools import *

import numpy as np
from numba import cuda, jit, vectorize, guvectorize


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
    
    nvars = len(ring.symbols)  # for striding
    lt_buf = np.zeros(nvars + 1, dtype=np.float32)
    lt_buf[-1] = 1  # lt's coeff is defined as ring.domain.one
    fdest = np.zeros(2 * nvars + 1, dtype=np.float32)
    gdest = np.zeros_like(fdest)

    f = get_cuda_cp_array(mod_pair[0], nvars)
    g = get_cuda_cp_array(mod_pair[1], nvars)

    kernel_data = [nvars, lt_buf, f, g, fdest, gdest]
    cuda_cp_arys = numba_cp_kernel_launch(kernel_data)

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
    cuda_cp_array = np.zeros(2 * nvars + 1, dtype=np.float32)
    
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
    lt_buf = kernel_data[1]
    f = kernel_data[2]
    g = kernel_data[3]
    fdest = kernel_data[4]
    gdest = kernel_data[5]

    # Launch kernel
    numba_critical_pairs(nvars, lt_buf, f, g, fdest, gdest)
    return [fdest, gdest]


@cuda.jit
def numba_critical_pairs(nvars, lt_buf, f, g, fdest, gdest):
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
    # Going to start by just writing out sequentially and seeing if numba
    # will vectorize or auto-jit it for us.

    # Monomial LCM
    for i in range(nvars):
        lt_buf[i] = max(f[nvars + i], g[nvars + i])

    # Calculate um, vm (data independent)
    for i in range(nvars):
        fdest[nvars + i] = lt_buf[i] - f[nvars + i]
        gdest[nvars + i] = lt_buf[i] - g[nvars + i]

    fdest[-1] = lt_buf[-1] / f[-1]
    gdest[-1] = lt_buf[-1] / g[-1]

    # Calculate Sign(fr), Sign(gr) (data independent)
    for i in range(nvars):
        fdest[i] = f[i] + fdest[nvars + i]
        gdest[i] = g[i] + gdest[nvars + i]


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
    domain = ring.domain
    nvars = len(ring.symbols)

    gpu_sympy_cp = []

    # Build critical pair list
    gpu_sympy_cp.append([cuda_cp[0][:nvars], int(pair[0][0][1])]) # sig(fr)
    gpu_sympy_cp.append([cuda_cp[0][nvars:-1], cuda_cp[0][-1]]) # um
    gpu_sympy_cp.append(pair[0]) # f
    gpu_sympy_cp.append([cuda_cp[1][:nvars], int(pair[1][0][1])]) # sig(gr)
    gpu_sympy_cp.append([cuda_cp[1][nvars:-1], cuda_cp[1][-1]]) # vm
    gpu_sympy_cp.append(pair[1]) # g

    # Type casting - sigs, exponents integers, coeffs depend on domain
    # Sign(fr), um exponents, Sign(gr), vm exponents
    gpu_sympy_cp[0][0] = tuple([int(s) for s in gpu_sympy_cp[0][0]])
    gpu_sympy_cp[1][0] = tuple([int(e) for e in gpu_sympy_cp[1][0]])
    gpu_sympy_cp[3][0] = tuple([int(s) for s in gpu_sympy_cp[3][0]])
    gpu_sympy_cp[4][0] = tuple([int(e) for e in gpu_sympy_cp[4][0]])
    
    if domain == QQ:
        # um, vm coefficient
        gpu_sympy_cp[1][1] = Rational(float(gpu_sympy_cp[1][1])).limit_denominator(10**6)
        gpu_sympy_cp[4][1] = Rational(float(gpu_sympy_cp[4][1])).limit_denominator(10**6)

    elif domain == RR:
        pass
    
    else:
        print("Fields other than QQ, RR not implemented on GPU at this time")
        sys.exit(0)

    # Retuple
    gpu_sympy_cp[0] = tuple(gpu_sympy_cp[0])
    gpu_sympy_cp[1] = tuple(gpu_sympy_cp[1])
    gpu_sympy_cp[3] = tuple(gpu_sympy_cp[3])
    gpu_sympy_cp[4] = tuple(gpu_sympy_cp[4])
        
    return tuple(gpu_sympy_cp)


if __name__ == "__main__":
    print("GPU Critical Pairs Test")
    print("Cyclic Affine 4")
    
    r, a, b, c, d = ring(symbols="a, b, c, d", domain="QQ", order="grevlex")
    f1 = a + b + c + d
    f2 = a*b + b*c + a*d + c*d
    f3 = a*b*c + a*b*d + a*c*d + b*c*d
    f4 = a*b*c*d - 1
    F = [f1, f2, f3, f4]

    B = [lbp(sig(r.zero_monom, i), f, i) for i, f in enumerate(F)]

    GPU_CP = [cp_cuda(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    for i, cp in enumerate(GPU_CP):
        print("GPU Critical Pair {}: ".format(i))
        for i, part in enumerate(cp):
            print(i, part)

    CP = [critical_pair(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]
    print('-------------------')
    for i, cp in enumerate(CP):
        print("Original Critical Pair {}:".format(i))
        for j, part in enumerate(cp):
            print(j, part)


################################################################################
def old_cuda_cp(B, ring):
    # Original Form:
    # CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]

    # LT format -> ((0, 1, 0, 0), 6):
    #   the first tuple represents the exponents for each possible term (x0, x1, x2, x3) Above, the tuple is x1 ** 2
    #   the last number represents the coefficient, so the whole tuple together represents 6x ** 2
    #
    # The Polyn(f).leading_term() below is similar, but in the format ' 6*x1**2 '
    #
    # Required Output format:
    #
    # [
    #   (Sign(gr), vm, g, Sign(fr), um, f),
    #   (Sign(gr), vm, g, Sign(fr), um, f)
    # ]

    cp_res = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            f = B[i]
            g = B[j]

            domain = ring.domain

            ltf = Polyn(f).LT
            ltg = Polyn(g).LT

            lt = []

            for a, b in zip(ltf[0], ltg[0]):
                lt.append(max(a, b))
            lt = (tuple(lt), domain.one)

            lt_lm, lt_lc = lt
            ltf_lm, ltf_lc = ltf

            C = tuple([a - b for a, b in zip(lt_lm, ltf_lm)])

            if all(c >= 0 for c in C):
                monom = tuple(C)
            else:
                monom = None

            if domain.is_Field:
                if monom is not None:
                    # um = monom, domain.quo(lt_lc, ltf_lc)
                    um = monom, lt_lc / ltf_lc
                else:
                    um = None
            else:
                print('ERROR - Non-Field domains not supported in CUDA version')
                exit()
                # if not (monom is None or lt_lc % ltf_lc):
                #     um = monom, domain.quo(lt_lc, ltf_lc)
                # else:
                #     um = None

            lt_lm, lt_lc = lt
            ltg_lm, ltg_lc = ltg

            C = tuple([a - b for a, b in zip(lt_lm, ltg_lm)])

            if all(c >= 0 for c in C):
                monom = tuple(C)
            else:
                monom = None

            if domain.is_Field:
                if monom is not None:
                    vm = monom, lt_lc / ltg_lc
                else:
                    vm = None
            else:
                print('ERROR - Non-field domains not supported in CUDA version')
                # if not (monom is None or lt_lc % ltg_lc):
                #     vm = monom, domain.quo(lt_lc, ltg_lc)
                # else:
                #     vm = None
            f2 = tuple(Polyn(f))
            fr = (
                tuple((tuple([a + b for a, b in zip(Sign(f)[0], um[0])]), Sign(f)[1])),
                Polyn(f).mul_term(um),  # This guy is the last big issue
                Num(f)
            )

            def mul_poly_by_term(f, term):
                """
                Local term * poly multiplication function.
                Takes a PolyElement f, and tuple term
                Returns a list of term tuples representing a polynomial

                Ex: mul_poly_by_term(f1, ((0, 0, 0), 1))
                """
                new_poly = []
                mono_mul = lambda m1, m2: tuple([a + b for a, b in zip(m1, m2)])
                for m, c in f.terms():
                    monom = mono_mul(term[0], m)
                    coeff = c * term[1]
                    new_poly.append((monom, coeff))
                return new_poly

            def mul_tlist_by_term(tlist, term):
                """
                term * poly for polynomial term list form
                In: tlist: [((monom), coeff)...], term: ((monom), coeff)
                Out: [((monom), coeff)...]
                """
                mono_mul = lambda m1, m2: tuple([a + b for a, b in zip(m1, m2)])
                new_poly = []
                for m, c in tlist:
                    monom = mono_mul(term[0], m)
                    coeff = c * term[1]
                    new_poly.append((monom, coeff))
                return new_poly

            def term_list_to_sympy(tlist, ring):
                """
                Given a polynomial as a list of term tuples
                return a PolyElement object in the given ring.

                Ex: tlist = [((1, 2, 3), 1), ((2, 0, 4), 3))]
                sympy_poly = term_list_to_sympy(tlist, ring)
                >>> x*y**2*z***3 + 3*x**2*z**4
                >>> type(sympy_poly) -> PolyElement
                """
                pexp = []
                for m, c in tlist:
                    pexp.append('+' + str(c))
                    for e, s in zip(m, ring.symbols):
                        pexp.append('*' + str(s) + '**' + str(e))
                return ring.from_expr(''.join(pexp))

            # def mul_term(f, term):
            #     monom, coeff = term
            #
            #     if not f or not coeff:
            #         return f.ring.zero
            #     elif monom == f.ring.zero_monom:
            #         return f.mul_ground(coeff)
            #
            #     monomial_mul = f.ring.monomial_mul
            #     terms = [(monomial_mul(f_monom, monom), f_coeff * coeff) for f_monom, f_coeff in f.items()]
            #     return f.new(terms)

            gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)

            # this just returns in the correct order, so should not need to be parallelized I think
            if lbp_cmp(fr, gr) == -1:
                cp_res.append((Sign(gr), vm, g, Sign(fr), um, f))
            else:
                cp_res.append((Sign(fr), um, f, Sign(gr), vm, g))

    return cp_res

