"""Convert KEEL-style .dat datasets to .csv files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

def convert_dat_to_csv(dat_path: str, csv_path: str | None = None) -> Path:
    attributes = []
    data_lines = []
    in_data_section = False

    dat_file = Path(dat_path)
    if csv_path is None:
        csv_file = dat_file.with_suffix('.csv')
    else:
        csv_file = Path(csv_path)

    with dat_file.open('r', encoding='utf-8') as f:
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

    # Attempt numeric conversion per column while preserving non-numeric labels.
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            # Keep categorical/text columns unchanged.
            pass

    # Save to CSV
    os.makedirs(csv_file.parent, exist_ok=True)
    df.to_csv(csv_file, index=False)
    print(f"Saved {len(df)} rows to: {csv_file}")
    print(f"Columns: {list(df.columns)}")
    if 'Class' in df.columns:
        print(f"Class distribution:\n{df['Class'].value_counts()}")
    return csv_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert KEEL .dat files to CSV.')
    parser.add_argument('dat_path', nargs='+', help='Input .dat file path(s).')
    parser.add_argument('--output', '-o', help='Optional output CSV path (single input only).')
    args = parser.parse_args()

    if args.output and len(args.dat_path) != 1:
        raise ValueError('--output can only be used with a single input file.')

    for in_path in args.dat_path:
        out_path = args.output if len(args.dat_path) == 1 else None
        convert_dat_to_csv(in_path, out_path)
