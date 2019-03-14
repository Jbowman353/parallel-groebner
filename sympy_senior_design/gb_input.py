from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ

x1 = 0
x2 = 0
x5 = 0
x4 = 0
x3 = 0
inputs = [{'file': 'cyclic_affine_5.txt', 'r_v': xring(['x1', 'x2', 'x3', 'x4', 'x5'], QQ, lex), 'system': [x1+x2+x3+x4+x5, x1*x2+x2*x3+x3*x4+x1*x5+x4*x5, x1*x2*x3+x2*x3*x4+x1*x2*x5+x1*x4*x5+x3*x4*x5, x1*x2*x3*x4+x1*x2*x3*x5+x1*x2*x4*x5+x1*x3*x4*x5+x2*x3*x4*x5, x1*x2*x3*x4*x5-1]}, {'file': 'katsura_affine_5.txt', 'r_v': xring(['x1', 'x2', 'x3', 'x4', 'x5'], QQ, lex), 'system': [x1+2*x2+2*x3+2*x4+2*x5-1, x1**2+2*x2**2+2*x3**2+2*x4**2+2*x5**2-x1, 2*x1*x2+2*x2*x3+2*x3*x4+2*x4*x5-x2, x2**2+2*x1*x3+2*x2*x4+2*x3*x5-x3, 2*x2*x3+2*x1*x4+2*x2*x5-x4]}, ]
