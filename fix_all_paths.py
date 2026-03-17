import json
import glob
import os

files = glob.glob('notebook_modified/*.ipynb')

for fp in files:
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        modified = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                for i, line in enumerate(source):
                    if '/home/quangvd/project/FAIR-2022/' in line:
                        source[i] = line.replace('/home/quangvd/project/FAIR-2022/', '../')
                        modified = True
                        
                # Also deduplicate ir_imb if present
                new_source = []
                ir_count = 0
                cell_modified = False
                for line in source:
                    if 'ir_imb =' in line:
                        if ir_count == 0:
                            new_source.append(line)
                        else:
                            cell_modified = True
                        ir_count += 1
                    elif 'print(f"IR (neg/pos)' in line:
                        if ir_count <= 1:
                            new_source.append(line)
                        else:
                            cell_modified = True
                    else:
                        new_source.append(line)
                
                if cell_modified:
                    cell['source'] = new_source
                    modified = True
                
        if modified:
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f'Fixed paths in {fp}')
    except Exception as e:
        print(f"Error on {fp}: {e}")
