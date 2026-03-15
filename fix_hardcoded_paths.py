#!/usr/bin/env python3
"""
Script to fix hardcoded Windows paths in Processing_Data files
"""
import os
import re
from pathlib import Path

# Files that need fixing
files_to_fix = [
    'Co_Author_100_500_ir.py',
    'Co_Author_100_500.py',
    'Co_Author_100_700.py',
    'Co_Author_100_900.py',
    'CoAuthor_1800.py',
    'Co_Author_200_1000_1.py',
    'Co_Author_200_1000_ir.py',
    'Co_Author_200_1000.py',
    'Co_Author_250_750.py',
    'Co_Author_300_1500_ir.py',
    'Co_Author_300_1500.py',
    'Co_Author_50_250_ir.py',
    'Co_Author_50_250.py',
    'Co_Author_50_350.py',
    'CoAuthor_600.py',
    'Co_Author.py',
    'Co_Author_TestSize.py',
    'Ecoli_All.py',
    'Waveform_KFold.py',
]

base_dir = '/home/quangvd/project/FAIR-2022/Processing_Data'

def fix_file(filepath):
    """Fix hardcoded paths in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: data/datasets files (for Co_Author files)
    # D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/data/datasets/
    pattern1 = r"pd\.read_csv\('D:/[^']*?/data/datasets/([^']+)'\)"
    
    def replace_data_datasets(match):
        filename = match.group(1)
        return f"pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', '{filename}'))"
    
    content = re.sub(pattern1, replace_data_datasets, content)
    
    # Pattern 2: Processing_Data/dataset files
    # D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/FUZZY SVM/CODE/07_04_2022/fuzzy_svm/Processing_Data/dataset/
    pattern2 = r"pd\.read_csv\('D:/[^']*?/Processing_Data/dataset/([^']+)'\)"
    
    def replace_processing_data(match):
        filename = match.group(1)
        return f"pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', '{filename}'))"
    
    content = re.sub(pattern2, replace_processing_data, content)
    
    # Add import if needed and not already there
    if ("os.path.join" in content) and ("import os" not in content):
        # Add import os after other imports
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_idx = i + 1
        
        if insert_idx > 0:
            lines.insert(insert_idx, 'import os')
            content = '\n'.join(lines)
    
    # Only write if changes were made
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Fix all files
print("Fixing hardcoded paths in Processing_Data files...")
fixed_count = 0
for filename in files_to_fix:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        if fix_file(filepath):
            print(f"✓ Fixed: {filename}")
            fixed_count += 1
        else:
            print(f"✗ No changes: {filename}")
    else:
        print(f"✗ Not found: {filename}")

print(f"\nTotal files fixed: {fixed_count}/{len(files_to_fix)}")
