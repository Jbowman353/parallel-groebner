from sympy.polys.groebnertools import *
# from numba import cuda

# from cuda_cp_gf import cp_cuda
from cuda_cp_65521 import cp_cuda
from cuda_spoly_65521 import cuda_s_poly, cuda_s_poly2
# from cuda_spoly_32003 import cuda_s_poly2


##########################################################
#    GPU-based f5b
#
##########################################################

def _f5b_gpu(F, r, useGPUCP, useGPUSPoly):

    # select applicable function to use

    sp_needs_ring = False  # to decide whether or not to pass ring to s_poly (used in GPU one)

    if useGPUCP:
        c_p = cp_cuda
    else:
        c_p = critical_pair
    if useGPUSPoly:
        s_p = cuda_s_poly
        sp_needs_ring = True
    else:
        s_p = s_poly

    domain, orig = r.domain, None

    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, r = r, r.clone(domain=domain.get_field())
        except DomainError:
            raise DomainError("can't compute a Groebner basis over %s" % domain)
        else:
            F = [ s.set_ring(r) for s in F ]

    order = r.order

    # reduce polynomials (like in Mario Pernici's implementation) (Becker, Weispfenning, p. 203)
    B = F
    while True:
        F = B
        B = []

        for i in range(len(F)):
            p = F[i]
            rem = p.rem(F[:i])

            if rem:
                B.append(rem)

        if F == B:
            break

    # basis
    B = [lbp(sig(r.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)

    # critical pairs
    CP = [critical_pair(B[i], B[j], r) for i in range(len(B)) for j in range(i + 1, len(B))]

    CP.sort(key=lambda cp: cp_key(cp, r), reverse=True)

    k = len(B)

    reductions_to_zero = 0

    while len(CP):
        cp = CP.pop()

        # discard redundant critical pairs:
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue

        if sp_needs_ring:
            s = s_p(cp, B, r)
        else:
            s = s_p(cp)

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
                    cp = c_p(p, g, r)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue

                    CP.append(cp)

            # sort (other sorting methods/selection strategies were not as successful)
            CP.sort(key=lambda cp: cp_key(cp, r), reverse=True)

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
    H = red_groebner(H, r)

    return sorted(H, key=lambda f: order(f.LM), reverse=True)


def run(I, R, useGPUCP, useGPUSPoly):
    return _f5b_gpu(I, R, useGPUCP, useGPUSPoly)
