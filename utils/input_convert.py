import os

# use this for inputs from the gb package (https://github.com/ederc/gb/) or similarly formated ones
# Basically, replace all ^ with **, dump the lines into an array of strings
import_string = "from sympy.polys.orderings import lex, grlex\nfrom sympy.polys.rings import ring, xring\nfrom sympy.polys.domains import ZZ, QQ\n\n"

def convert_gb_input(filepath, outfile=None):
    variables = []
    newInputs = []
    with open(filepath) as gb_file:
        variables = str(gb_file.readline()).replace('\n', '').strip().split(',') #First line
        variables = [x.strip() for x in variables]
        gb_file.readline() #ignore second line
        newInputs = [str(line).replace('^', '**').replace('\n', '').replace(',','') for line in gb_file]
    output = {'vars': variables, 'system': newInputs}

    if outfile:
        with open(outfile, 'w') as out_file:
            out_file.write(','.join(output['vars']) + '\n')
            for p in output['system']:
                if output['system'][-1] is p:
                    out_file.write(p)
                else:
                    out_file.write(p + ',\n')

    else:
        print(output)

    return output

def convertAllGBInputsToPython():
    systems = []
    vars_to_define = set([])
    to_write = []
    for filename in os.listdir('../test_inputs_from_gb'):
        systems.append({
            'file': filename,
            'data': convert_gb_input('../test_inputs_from_gb/' + filename)
        })

    with open('gb_input.py', 'w') as new_input_file:
        for system in systems:
            fname = system['file']
            psys = str(system['data']['system']).replace(r"'", '')
            variables = str(system['data']['vars'])

            vars_to_define = vars_to_define.union(system['data']['vars'])

            xr_data = variables + ', QQ, lex'

            to_write.append(r"{'file': '" + fname + r"', 'r_v': xring(" + xr_data + r"), 'system': " + psys + "}, ")
        

        new_input_file.write(import_string)
        for v in vars_to_define:
            new_input_file.write(v + " = 0\n")
        new_input_file.write('inputs = [')
        for l in to_write:
            new_input_file.write(l)
        new_input_file.write(']\n')