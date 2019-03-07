#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re


def parse_polynomial(poly_string):
    """
    Parses a polynomial string.
    Takes a string formatted for example as "3x1^2x2^3+4x3^4" etc.
    Uses a compiled regex pattern. The included pattern is for the above representation.
    Returns a list of dictionaries representing one polynomial in the system.
    """

    # Read poly string into list of dictionaries of all variable components.
    pattern = r"(([+-]?\d*)?(?:x(\d+)(?:\^(\d+))?))?|([+-]\d+)"
    prog = re.compile(pattern)
    result = prog.findall(poly_string)
    poly_parts = []

    for match in result:
        term_part = dict()
        if match[0] == '':
            continue
        else:
            term_part["term"] = match[0]

        if match[1] == '':
            term_part["coeff"] = 1
        else:
            term_part["coeff"] = int(match[1])

        if match[3] == '':
            term_part[f"x{match[2]}"] = 1
        else:
            term_part[f"x{match[2]}"] = int(match[3])

        poly_parts.append(term_part)

    # Reconstruct monomials from poly_parts
    new_monomials = [0] * len(poly_parts)
    poly_list = []

    for i in range(len(poly_parts)):
        tag = poly_parts[i]["term"][0]

        if (tag == '+') or (tag == '-') or (i == 0):
            new_monomials[i] = 1

    for i in range(sum(new_monomials)):
        poly_list.append(dict())

    cur_term = 0
    for i in range(len(new_monomials)):
        cur_term += new_monomials[i]
        if new_monomials[i] == 1:
            poly_list[cur_term-1] = {**poly_parts[i],
                                     **poly_list[cur_term-1]}
            j = i + 1
            try:
                while new_monomials[j] != 1:
                    poly_list[cur_term-1] = {**poly_parts[j],
                                             **poly_list[cur_term-1]}
                    j += 1
            except:
                continue

    # Collect all variables present in polynomial and represent in each monomial
    variables = []

    for p in poly_list:
        for k, v in p.items():
            if k == "term":
                p["term"] = ''
                continue
            if k[0] == 'x':
                p["term"] += f"{k}^{v}"
                if k not in variables:
                    variables.append(k)

    variables = sorted(variables)

    for v in variables:
        for p in poly_list:
            if v not in p.keys():
                p[v] = 0

    # Add total degree of monomial
    for p in poly_list:
        p["total_degree"] = 0
        for v in variables:
            p["total_degree"] += p[v]

    # Add Python native representation
    python_poly_string = ""
    for p in poly_list:
        python_poly_string += '+' + str(p["coeff"])
        for v in variables:
            python_poly_string += '*' + str(v) + '**' + str(p[v])

    poly_list.append(python_poly_string)

    return poly_list
        
    
        
if __name__ == '__main__':
    p_string = "5x1^3x2^2+2x2^2x3+4x4^5x5^6"
    parsed = parse_polynomial(p_string)

    print(parsed[-1])
