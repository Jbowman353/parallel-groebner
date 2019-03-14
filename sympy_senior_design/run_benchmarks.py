import sympy
import gb_input
import csv, sys, os, datetime

print(gb_input.inputs)

methods = ['f5b', 'f5b_gpu']

output_headers = ['Input Name', 'SymPy F5B Time', 'F5B-Like GPU Time', 'Matching Results?']

output_data = []

output_file_path = 'bench_output_' + datetime.datetime.now().strftime("%H_%M__%b_%d") + '.csv'

