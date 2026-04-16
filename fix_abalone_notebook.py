import json

# Read file with proper encoding
with open('./notebook_modified/PSO_AFWCIL_QuickTest_abalone.ipynb', 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)

# First markdown cell - update description
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and len(cell.get('source', [])) > 0:
        src = ''.join(cell['source'])
        if 'haberman' in src or 'transfusion' in src:
            new_source = []
            for line in cell['source']:
                line = line.replace('haberman', 'abalone')
                line = line.replace('transfusion', 'abalone')
                new_source.append(line)
            cell['source'] = new_source
            break

# Find and update dataset configuration cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'DATASET_NAME' in src:
            # Replace configuration
            new_source = []
            for line in cell['source']:
                line = line.replace('DATASET_NAME = "transfusion"', 'DATASET_NAME = "abalone"')
                line = line.replace('DATASET_FILE = "../data/datasets/transfusion.csv"', 'DATASET_FILE = "../data/datasets/abalone.csv"')
                line = line.replace('whether he/she donated blood in March 2007', 'Class')
                line = line.replace('{1: 1.0, 0: -1.0}   # donate=1', "{'positive': 1.0, 'negative': -1.0}   # positive=1")
                line = line.replace('CAT_COLS     = []', "CAT_COLS     = ['Sex']                 # Sex là categorical (M, F, I)")
                new_source.append(line)
            cell['source'] = new_source
            break

# Write back
with open('./notebook_modified/PSO_AFWCIL_QuickTest_abalone.ipynb', 'w', encoding='utf-8-sig') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('✔ File updated successfully')
