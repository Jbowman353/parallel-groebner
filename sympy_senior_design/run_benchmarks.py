from sympy.polys.groebnertools import groebner, is_groebner
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
import gb_input # take gb_input.py generated file from input_convert and make sure it's visible
import csv, sys, os, datetime

assert len(gb_input.inputs) != 0

methods = ['f5b', 'f5b_gpu']

output_headers = ['Input Name', 'SymPy F5B Time', 'F5B-Like GPU Time', 'Matching Results?']

output_data = []

output_file_path = 'bench_output_' + datetime.datetime.now().strftime("%H_%M__%b_%d") + '.csv'

for input in gb_input.inputs:
    fname = input['file']
    var_str_list = input['vars']
    sys_string = input['system']

    r_v = xring(var_str_list, QQ, lex)

    var_list = []
    for i in range(len(var_str_list)):
        new_var = r_v[1][i]
        exec(var_str_list[i] + ' = ' + 'new_var')
        exec('var_list.append({})'.format(var_str_list[i]))
    
    I = None
    R = r_v[0]

    exec('I = ' + sys_string)

    print(is_groebner(groebner(I, R), R))

