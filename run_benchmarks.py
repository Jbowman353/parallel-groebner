from sympy.polys.groebnertools import groebner, is_groebner
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ

import f5b_gpu
import gb_input # take gb_input.py generated file from input_convert and make sure it's visible

import time, csv, sys, os, datetime

assert len(gb_input.inputs) != 0

print('*****\nRUNNING BENCHMARKS\n******')

output_headers = ['Input Name', 'SymPy F5B Time', 'F5B-Like GPU Time', 'Matching Results?', 'Fastest Runtime']

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

    start_f5b = time.time()
    res_f5b = groebner(I, R, method="f5b")
    end_f5b = time.time()
    f5b_runtime = end_f5b - start_f5b

    # start_f5b_gpu = time.time()
    # res_f5b_gpu = f5b_gpu.run(I, R)
    # end_f5b_gpu = time.time()
    # f5b_gpu_runtime = end_f5b_gpu - start_f5b_gpu

    output_data.append({'Input Name': fname, 'SymPy F5B Time': f5b_runtime, 'F5B-Like GPU Time': None, 'Matching Results?': False, 'Fastest Runtime':None})
    

with open(output_file_path, 'w') as csv_file:
    out_writer = csv.DictWriter(csv_file, fieldnames = output_data[0].keys())
    out_writer.writeheader()
    out_writer.writerows(output_data)