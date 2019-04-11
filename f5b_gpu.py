from sympy.polys.groebnertools import *
# from numba import cuda
import numba
import numpy
import math


##########################################################
#    GPU-based f5b
#    *As of now, I've just copied the contents of _f5b above, and possibly made minor changes, do not assume this works
##########################################################

def _f5b_gpu(F, ring):
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

# customized spoly with localized functions
# @numba.vectorize(['float32(float32, float32, float32)'], target='cuda')
def cuda_spoly(p1, p2, ring):
    """
    Compute LCM(LM(p1), LM(p2))/LM(p1)*p1 - LCM(LM(p1), LM(p2))/LM(p2)*p2
    This is the S-poly provided p1 and p2 are monic
    """
    LM1 = p1.LM
    LM2 = p2.LM

    # rewrite of LCM12 = custom
    LCM12 = tuple([max(a, b) for a, b in zip(LM1, LM2)])

    # rewrite of m1 = custom_monomial_div(LCM12, LM1)
    C = tuple([a - b for a, b in zip(LCM12, LM1)])
    if all(c >= 0 for c in C):
        m1 = tuple(C)

    # rewrite of m2 = custom_monomial_div(LCM12, LM2)
    C = tuple([a - b for a, b in zip(LCM12, LM2)])
    if all(c >= 0 for c in C):
        m2 = tuple(C)

    # rewrite of s1 = custom_mul_monom(p1,m1)
    monomial_mul = p1.ring.monomial_mul
    terms = [(tuple([a + b for a, b in zip(f_monom, m1)]), f_coeff) for f_monom, f_coeff in p1.items()]
    s1 = p1.new(terms)

    # rewrite of s2 = custom_mul_monom(p2,m2)
    monomial_mul = p2.ring.monomial_mul
    terms = [(tuple([a + b for a, b in zip(f_monom, m2)]), f_coeff) for f_monom, f_coeff in p2.items()]
    s2 = p2.new(terms)

    s = s1 - s2
    return s


def cuda_cp(B, ring):
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
        for j in range(i+1, len(B)):
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

            C = tuple([ a - b for a, b in zip(lt_lm, ltf_lm) ])

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

            C = tuple([ a - b for a, b in zip(lt_lm, ltg_lm) ])

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

            fr = (
                tuple((tuple([a + b for a, b in zip(Sign(f)[0], um[0])]), Sign(f)[1])),
                Polyn(f).mul_term(um), # This guy is the last big issue
                Num(f)
            )


            gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)

            #this just returns in the correct order, so should not need to be parallelized I think
            if lbp_cmp(fr, gr) == -1:
                cp_res.append((Sign(gr), vm, g, Sign(fr), um, f))
            else:
                cp_res.append((Sign(fr), um, f, Sign(gr), vm, g))

    return cp_res


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


def run(I, R):
    return _f5b_gpu(I, R)
