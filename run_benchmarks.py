from sympy.polys.groebnertools import groebner, is_groebner
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ, RR

import f5b_gpu
import gb_input # take gb_input.py generated file from input_convert and make sure it's visible

import time, csv, sys, os, datetime, re

def get_degree(sys_string):
    newstr = sys_string.replace(' ', '')
    degs = []

    for i in range(len(newstr)):
        if newstr[i] == '*' and newstr[i+1] == '*':
            j = i + 2
            deg = newstr[j]  # get first number of exponent
            while True:
                j += 1
                # Check to make sure its a number
                try:
                    float(newstr[j])
                except ValueError:
                    break
                deg += newstr[j]
            degs.append(deg)

    return max(degs) if len(degs) != 0 else 1


assert len(gb_input.inputs) != 0

print('*****\nRUNNING BENCHMARKS\n******')

output_data = []

output_file_path = 'bench_output_' + datetime.datetime.now().strftime("%H_%M__%b_%d") + '.csv'

gb_input.inputs.sort(key=lambda x: len(x['vars']))

for input_data in gb_input.inputs:
    fname = input_data['file']
    var_str_list = input_data['vars']
    sys_string = input_data['system']

    r_v = xring(var_str_list, RR, lex)
    # r_v = xring(var_str_list, QQ, lex)

    var_list = []
    for i in range(len(var_str_list)):
        new_var = r_v[1][i]
        exec(var_str_list[i] + ' = ' + 'new_var')
        exec('var_list.append({})'.format(var_str_list[i]))
    
    I = None
    R = r_v[0]

    exec('I = ' + sys_string)

    print(fname + ' Starting ...')
    print(I)
    print(R)

    start_f5b = time.time()
    res_f5b = groebner(I, R, method="f5b")
    end_f5b = time.time()
    f5b_runtime = end_f5b - start_f5b

    start_cp_gpu = time.time()
    res_cp_gpu = f5b_gpu.run(I, R, True, False)  # Only GPU CP
    end_cp_gpu = time.time()
    cp_gpu_runtime = end_cp_gpu - start_cp_gpu

    #######
    #
    #   HAVE NOT SET UP CP, SP, CP+SP CASES YET
    #######

    # start_sp_gpu = time.time()
    # res_sp_gpu = f5b_gpu.run(I, R, False, True)  # Only GPU Spoly
    # end_sp_gpu = time.time()
    # sp_gpu_runtime = end_sp_gpu - start_sp_gpu
    #
    # start_cpsp_gpu = time.time()
    # res_cpsp_gpu = f5b_gpu.run(I, R, True, True)  # Both CP and SPoly
    # end_cpsp_gpu = time.time()
    # cpsp_gpu_runtime = end_cpsp_gpu - start_cpsp_gpu



    output_data.append({
        'Input Name': fname,
        'SymPy F5B Time (sec)': f5b_runtime,
        'CP GPU Time (sec)': cp_gpu_runtime,
        # 'SP GPU Time (sec)': sp_gpu_runtime,
        # 'CP+SP GPU Time (sec)': cpsp_gpu_runtime,
        'Number of Variables': len(var_list),
        'Number of Polynomials': len(I),
        'Max Degree': get_degree(sys_string)
    })

    print(fname + ' Completed!\n ----------')
    

with open(output_file_path, 'w') as csv_file:
    out_writer = csv.DictWriter(csv_file, fieldnames=output_data[0].keys())
    out_writer.writeheader()
    out_writer.writerows(output_data)