import os

# use this for inputs from the gb package (https://github.com/ederc/gb/) or similarly formated ones
# Basically, replace all ^ with **, dump the lines into an array of strings
def convert_gb_input(filepath, outfile=None):
    variables = []
    newInputs = []
    with open(filepath) as gb_file:
        variables = str(gb_file.readline()).replace('\n', '').strip().split(',') #First line
        gb_file.readline() #ignore second line
        newInputs = [str(line).replace('^', '**').replace('\n', '') for line in gb_file]
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

