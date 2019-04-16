from sympy.polys.groebnertools import *
# from numba import cuda
from cuda_cp import cp_cuda


##########################################################
#    GPU-based f5b
#
##########################################################

def _f5b_gpu(F, ring, useGPUCP, useGPUSPoly):
    if useGPUCP:
        c_p = cp_cuda
    else:
        c_p = critical_pair
    if useGPUSPoly:
        s_p = cuda_spoly
    else:
        s_p = spoly

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
    CP2 = [cp_cuda(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]

    if CP != CP2:
        print("CP NOT EQUAL TO CUDA CP")
        mismatches = [(x, y) for x, y in zip(CP, CP2) if x != y]
        for x, y in mismatches:
            print('--------------')
            print(str(x) + '\n' + str(y))
            print('--------------')
            print('--------------')

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


def run(I, R, useGPUCP, useGPUSPoly):
    return _f5b_gpu(I, R, useGPUCP, useGPUSPoly)
