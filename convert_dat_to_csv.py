"""
Script to convert abalone19.dat (KEEL format) to CSV
Output: data/datasets/abalone19.csv
"""
import os
import pandas as pd

def convert_dat_to_csv(dat_path, csv_path):
    attributes = []
    data_lines = []
    in_data_section = False

    with open(dat_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('@attribute'):
                # Parse attribute name
                parts = line.split()
                attr_name = parts[1]
                attributes.append(attr_name)
            elif line.lower().startswith('@data'):
                in_data_section = True
            elif in_data_section:
                data_lines.append(line)

    # Parse data rows
    rows = []
    for line in data_lines:
        values = [v.strip() for v in line.split(',')]
        rows.append(values)

    df = pd.DataFrame(rows, columns=attributes)

    # Convert numeric columns (all except Sex and Class)
    numeric_cols = ['Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Save to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    return df

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    dat_path = os.path.join(base, 'data', 'datasets', 'dat-file', 'abalone19.dat')
    csv_path = os.path.join(base, 'data', 'datasets', 'abalone19.csv')
    convert_dat_to_csv(dat_path, csv_path)
