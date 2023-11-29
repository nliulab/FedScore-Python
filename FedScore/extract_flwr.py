import os, sys
import pandas as pd

def extract_coefficient(path):
    # print(path)
    dir, filename = os.path.dirname(path), os.path.basename(path)
    with open(path, 'r') as f:
        lines = f.readlines()
        coef = []
        for i, line in enumerate(lines):
            if not line:
                continue
            elif "printing final parameters" in line:
                start = 1
                while not 'array' in lines[i + start]:
                    start += 1
                end = 1
                while not (')]' in lines[i+end] and 'array' in lines[i + end]):
                    end += 1
                coef = lines[i + start:i + end + 1]
                coef = [x.strip().strip('\n') for x in coef]
                coef = eval(''.join(coef).replace('array', '').replace('(', '').replace(')', ''))
            elif '===========' in line and '=========' in lines[i + 2]:
                variables = eval(lines[i + 1])

    coef = coef[0][0]
    result = pd.Series(data=coef, index=variables, name='coef')
    out_filename = 'coef_' + '_'.join(filename.split('.')[0].split('_')[1:]) + '.csv'
    result.to_csv(os.path.join(in_path, out_filename))


if __name__ == "__main__":
    in_path = sys.argv[1]
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file.endswith('.txt'):
                extract_coefficient(os.path.join(root, file))
