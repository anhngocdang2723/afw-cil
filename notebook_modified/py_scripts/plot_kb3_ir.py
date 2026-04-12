import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime
import re

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 31,
    'axes.titlesize': 34,
    'axes.labelsize': 31,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 25,
    'figure.titlesize': 34,
})


def _resolve_default_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'notebook_modified', 'Experiment'),
        os.path.join(script_dir, 'Experiment'),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def plot_kb3_ir_all_datasets(base_dir=None, metrics=None, dataset_order=None):
    if base_dir is None:
        base_dir = _resolve_default_base_dir()

    if dataset_order is None:
        dataset_order = ['abalone','haberman', 'vehicle1', 'transfusion', 'bankrupt']

    if metrics is None:
        metrics = ['Gmean_mean', 'SE_mean', 'SP_mean', 'AUC_mean']

    default_ir_levels = [0.08, 0.06, 0.04, 0.02]

    def _format_ir_percent_label(value):
        percent = value * 100
        if abs(percent - round(percent)) < 1e-9:
            return f"{int(round(percent))}%"
        return f"{percent:.1f}%"

    dataset_aliases = {
        'bankrupt': ['bankrupt', 'company_bankruptcy_rediction'],
    }

    alias_to_target = {}
    for ds_name in dataset_order:
        alias_to_target[ds_name] = ds_name
    for target_name, aliases in dataset_aliases.items():
        if target_name in dataset_order:
            for alias in aliases:
                alias_to_target[alias] = target_name
        
    # Tạo folder chung lưu kết quả theo mốc thời gian để khỏi đè nhau
    timestamp_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_dir = os.path.join(base_dir, f"KB3_Charts_{timestamp_str}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Tìm tất cả file KB3_summary
    # Ta sẽ dùng các file _extended.csv để đảm bảo có đủ các độ đo (Gmean_mean, SE_mean, v.v.)
    # Nếu không có extended, ta dùng file mặc định. Nên ta lọc ra lấy file mới nhất cho từng dataset
    all_summary_files = glob.glob(os.path.join(base_dir, "Test_*", "*KB3_summary*.csv"))
    if not all_summary_files:
        print(f"Warning: Không tìm thấy file KB3_summary nào trong thư mục: {base_dir}")
    
    # Gom file theo dataset
    dataset_files = {}
    for f in all_summary_files:
        # Tên file dạng: Test_wisconsin_KB3_summary_23032026_032024.csv 
        # hay Test_wisconsin_KB3_summary_23032026_032024_extended.csv
        file_name = os.path.basename(f)
        match = re.match(r"Test_([a-zA-Z0-9_]+)_KB3_summary", file_name)
        if match:
            source_ds_name = match.group(1)
            ds_name = alias_to_target.get(source_ds_name)
            if ds_name is None:
                continue
            if ds_name not in dataset_files:
                dataset_files[ds_name] = []
            dataset_files[ds_name].append(f)

    latest_files = {ds_name: sorted(files)[-1] for ds_name, files in dataset_files.items()}

    methods = ['WSVM', 'AFW', 'Proposed']
    colors = ['#1f77b4', '#d35400', '#2e7d32']
    markers = ['D', 's', 'x']
    line_styles = ['-', '-', '-.']
    y_ticks = np.arange(0.0, 1.01, 0.2)
    y_tick_labels = [f"{v:.1f}" for v in y_ticks]

    metric_labels = {
        'gmean': 'G-mean',
        'se': 'SE',
        'sp': 'SP',
        'auc': 'AUC',
    }

    def _metric_display_label(metric_name):
        metric_lower = metric_name.lower()
        for key, label in metric_labels.items():
            if key in metric_lower:
                return label
        return metric_name.replace('_mean', '').replace('_', ' ')

    for metric in metrics:
        dataset_plot_data = {}

        for ds_name in dataset_order:
            if ds_name not in latest_files:
                print(f"Warning: Không tìm thấy file KB3 cho dataset {ds_name}")
                continue

            latest_file = latest_files[ds_name]
            try:
                df = pd.read_csv(latest_file)
                ir_source_col = 'IR_target'

                if ds_name != 'bankrupt' and 'IR_target' in df.columns:
                    df = df[df['IR_target'].isin(default_ir_levels)].copy()
                    if not df.empty:
                        df['IR_target_rank'] = pd.Categorical(
                            df['IR_target'],
                            categories=default_ir_levels,
                            ordered=True,
                        )
                        # Keep IR in descending order: 8% -> 6% -> 4% -> 2%
                        df = df.sort_values('IR_target_rank', ascending=True)
                        df = df.drop(columns=['IR_target_rank'])
                elif ds_name == 'bankrupt':
                    if 'IR_actual' in df.columns:
                        ir_source_col = 'IR_actual'
                        df = df.sort_values('IR_actual', ascending=False)
                    else:
                        print("Warning: Dataset bankrupt không có cột IR_actual, dùng IR_target để hiển thị")

                actual_metric_cols = []
                for meth in methods:
                    if f"{meth}_{metric}" in df.columns:
                        actual_metric_cols.append(f"{meth}_{metric}")
                    elif f"{meth}_Gm_mean" in df.columns and 'Gmean' in metric:
                        actual_metric_cols.append(f"{meth}_Gm_mean")

                if len(actual_metric_cols) != 3:
                    print(f"Warning: Không đủ cột metric cho {ds_name} ở file {latest_file}. Cần metric: {metric}")
                    continue

                if df.empty:
                    print(f"Warning: Không có dữ liệu IR hợp lệ cho dataset {ds_name} sau khi lọc mốc IR")
                    continue

                ir_values = df[ir_source_col].to_numpy(dtype=float)
                y_by_method = {}
                for idx, meth in enumerate(methods):
                    y_by_method[meth] = df[actual_metric_cols[idx]].to_numpy(dtype=float)

                dataset_plot_data[ds_name] = {
                    'ir_values': ir_values,
                    'ir_labels': [_format_ir_percent_label(x) for x in ir_values],
                    'y_by_method': y_by_method,
                }

            except Exception as e:
                print(f"Lỗi đọc dữ liệu {ds_name} ({metric}): {str(e)}")

        n_datasets = len(dataset_order)
        fig, axes = plt.subplots(1, n_datasets, figsize=(5.4 * n_datasets, 8.2), sharey=True)
        axes = np.atleast_1d(axes)

        for i, ds_name in enumerate(dataset_order):
            ax = axes[i]

            if ds_name not in dataset_plot_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', color='#555555')
                ax.set_title(ds_name.capitalize(), fontweight='bold')
                ax.set_xticks([])
                ax.set_ylim(0, 1.05)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels)
                ax.grid(axis='y', linestyle='-', alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                continue

            ir_values = dataset_plot_data[ds_name]['ir_values']
            ir_labels = dataset_plot_data[ds_name]['ir_labels']
            y_by_method = dataset_plot_data[ds_name]['y_by_method']
            x_indexes = np.arange(len(ir_values))

            for idx, meth in enumerate(methods):
                ax.plot(
                    x_indexes,
                    y_by_method[meth],
                    label=meth,
                    color=colors[idx],
                    marker=markers[idx],
                    linestyle=line_styles[idx],
                    linewidth=2.0,
                    markersize=7,
                )

            ax.set_title(ds_name.capitalize(), fontweight='bold')
            ax.set_xticks(x_indexes)
            ax.set_xticklabels(ir_labels, color='#555555')
            ax.set_ylim(0, 1.05)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)

            if i == 0:
                ax.set_ylabel(_metric_display_label(metric))
            else:
                ax.tick_params(axis='y', labelleft=False)

            ax.grid(axis='y', linestyle='-', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)
        fig.supxlabel('Imbalance Ratio (IR, %)', y=0.11, fontsize=25)

        fig.patch.set_facecolor('white')
        for ax in axes:
            ax.set_facecolor('white')

        plt.tight_layout(rect=(0.01, 0.08, 0.99, 0.92))

        save_path = os.path.join(out_dir, f'KB3_{len(dataset_order)}_Datasets_{metric}.png')
        plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

        # Lưu thêm biểu đồ riêng cho từng dataset
        for ds_name in dataset_order:
            if ds_name not in dataset_plot_data:
                continue

            ir_values = dataset_plot_data[ds_name]['ir_values']
            ir_labels = dataset_plot_data[ds_name]['ir_labels']
            y_by_method = dataset_plot_data[ds_name]['y_by_method']
            x_indexes = np.arange(len(ir_values))

            fig_single, ax_single = plt.subplots(figsize=(8.5, 7.5))

            for idx, meth in enumerate(methods):
                ax_single.plot(
                    x_indexes,
                    y_by_method[meth],
                    label=meth,
                    color=colors[idx],
                    marker=markers[idx],
                    linestyle=line_styles[idx],
                    linewidth=2.0,
                    markersize=7,
                )

            ax_single.set_ylabel(_metric_display_label(metric))
            ax_single.set_xlabel('Imbalance Ratio (IR, %)', labelpad=4)
            ax_single.set_xticks(x_indexes)
            ax_single.set_xticklabels(ir_labels, color='#555555')
            ax_single.set_ylim(0, 1.05)
            ax_single.set_yticks(y_ticks)
            ax_single.set_yticklabels(y_tick_labels)
            ax_single.grid(axis='y', linestyle='-', alpha=0.3)
            ax_single.spines['top'].set_visible(False)
            ax_single.spines['right'].set_visible(False)
            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)

            fig_single.patch.set_facecolor('white')
            ax_single.set_facecolor('white')
            plt.tight_layout(rect=(0, 0, 1, 0.92))

            single_path = os.path.join(out_dir, f'KB3_{ds_name}_{metric}.png')
            plt.savefig(single_path, dpi=300, format='png', bbox_inches='tight')
            plt.close(fig_single)
            print(f"Saved: {single_path}")

    print(f"\n>> Đã lưu toàn bộ ảnh KB3 vào: {out_dir}")

if __name__ == '__main__':
    plot_kb3_ir_all_datasets()
