from sympy.polys.groebnertools import *
from numba import cuda
import numpy
import math


##########################################################
#    GPU-based f5b
#    *As of now, I've just copied the contents of _f5b above, and possibly made minor changes, do not assume this works
##########################################################

def _f5b_gpu(F, ring):
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

    # critical pairs
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    CP2 = cuda_cp(B, ring)
    if CP != CP2:
        print("CP NOT EQUAL TO CUDA CP")
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

    k = len(B)

    reductions_to_zero = 0

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
            else:
                for i, q in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break

            k += 1

            # print(len(B), len(CP), "%d critical pairs removed" % len(indices))
        else:
            reductions_to_zero += 1

    # reduce Groebner basis:
    H = [Polyn(g).monic() for g in B]
    H = red_groebner(H, ring)

    return sorted(H, key=lambda f: order(f.LM), reverse=True)

@cuda.jit
def domain_field_helper(poly_lt_arr, cp_res_arr):
    # For each pair (ltf, ltg) in poly_lt_arr:
    #  lt = (monomial_lcm(ltf[0], ltg[0]), domain.one) , monomial_lcm -> return tuple([max(a,b) for a,b in zip(A,B)])
    #  um = term_div(lt, ltf, domain)
    #  vm = term_div(lt, ltg, domain)
    #  fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    #  gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)
    #  * Then, return in correct order

    index_one = 0   # both indices here need fixed, just placeholders for now
    index_two = 1

    lt = []

    for a, b in zip(poly_lt_arr[index_one][0], poly_lt_arr[index_two][0]):
        lt.append(max(a, b))

    lt = tuple(lt)

    # TODO - um and vm calc


@cuda.jit
def domain_not_field_helper(poly_lt_arr, cp_res_arr):
    pass


def cuda_cp(B, ring):
    # Original Form:
    # CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]

    # LT format -> ((0, 1, 0, 0), 6):
    #   the first tuple represents the exponents for each possible term (x0, x1, x2, x3) Above, the tuple is x1 ** 2
    #   the last number represents the coefficient, so the whole tuple together represents 6x ** 2
    #
    # The Polyn(f).leading_term() below is similar, but in the format ' 6*x1**2 '

    poly_lt_arr = numpy.array([Polyn(f).LT for f in B])
    domain = ring.domain

    cp_res_arr = numpy.empty(math.factorial(poly_lt_arr.size))

    if domain.is_Field:
        domain_field_helper(poly_lt_arr, cp_res_arr)
    else:
        domain_not_field_helper(poly_lt_arr, cp_res_arr)

    # ltf = Polyn(f).LT
    # ltg = Polyn(g).LT
    # lt = (monomial_lcm(ltf[0], ltg[0]), domain.one) -> return tuple([ max(a, b) for a, b in zip(A, B) ])

    # um = term_div(lt, ltf, domain)
    # vm = term_div(lt, ltg, domain)
    # ---------------------------------
    #
    #        a_lm, a_lc = a
    #        b_lm, b_lc = b

    #        monom = monomial_div(a_lm, b_lm)

    #        if domain.is_Field:
    #            if monom is not None:
    #                return monom, domain.quo(a_lc, b_lc)
    #            else:
    #                return None
    #        else:
    #            if not (monom is None or a_lc % b_lc):
    #                return monom, domain.quo(a_lc, b_lc)
    #            else:
    #                return None
    #
    # ----------------------------------

    # # The full information is not needed (now), so only the product
    # # with the leading term is considered:
    # fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    # gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)

    # # return in proper order, such that the S-polynomial is just
    # # u_first * f_first - u_second * f_second:
    # if lbp_cmp(fr, gr) == -1:
    #     return (Sign(gr), vm, g, Sign(fr), um, f)
    # else:
    #     return (Sign(fr), um, f, Sign(gr), vm, g)


def run(I, R):
    return _f5b_gpu(I, R)