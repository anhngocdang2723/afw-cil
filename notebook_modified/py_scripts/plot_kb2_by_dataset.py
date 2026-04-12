import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'figure.titlesize': 22,
})

def plot_kb2_by_dataset(base_dir=None, dataset_order=None, metrics_to_plot=None):
    if base_dir is None:
        base_dir = r"d:\01_Study\Paper\paper_resources\01_src\afw-cil\notebook_modified\Experiment"
        
    if dataset_order is None:
        dataset_order = ['pima', 'haberman', 'vehicle3', 'yeast']
        
    if metrics_to_plot is None:
        # KB2 requested metrics: SE, SP, and Gmean
        metrics_to_plot = ['SE', 'SP', 'Gmean'] 
        
    methods_order = [
        'W-SVM',
        'BL-SMOTE+W-SVM', 
        'AFW-CIL_default',
        'BL-SMOTE+AFW',  
        'PSO-AFW-CIL', 
        'BL-SMOTE+PSO-AFW'
    ]

    # Alias để đọc được cả tên method cũ/mới trong các file summary khác nhau.
    method_aliases = {
        'W-SVM': ['W-SVM'],
        'BL-SMOTE+W-SVM': ['BL-SMOTE+W-SVM'],
        'AFW-CIL_default': ['AFW-CIL_default'],
        'BL-SMOTE+AFW': ['BL-SMOTE+AFW', 'BL-SMOTE+AFW-CIL_default'],
        'PSO-AFW-CIL': ['PSO-AFW-CIL'],
        'BL-SMOTE+PSO-AFW': ['BL-SMOTE+PSO-AFW', 'BL-SMOTE+PSO-AFW-CIL'],
    }
    
    # Store aggregated scores: dataset_name -> {metric: {method: score}}
    gathered_data = {ds: {m: {meth: 0.0 for meth in methods_order} for m in metrics_to_plot} for ds in dataset_order}

    # Extract Data from CSVs
    for ds_name in dataset_order:
        search_pattern = os.path.join(base_dir, f"Test_{ds_name}_*", f"*KB2_summary*.csv")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            continue
        
        latest_file = sorted(matching_files)[-1]
        
        try:
            df = pd.read_csv(latest_file)
            method_col = df['Name Method'].astype(str).str.strip()
            for meth in methods_order:
                aliases = method_aliases.get(meth, [meth])
                mask = pd.Series(False, index=df.index)

                # Ưu tiên exact-match theo alias, sau đó mới fallback contains.
                for alias in aliases:
                    exact_mask = method_col == alias
                    if exact_mask.any():
                        mask = exact_mask
                        break

                if not mask.any():
                    for alias in aliases:
                        contains_mask = method_col.str.contains(alias, na=False, regex=False)
                        if contains_mask.any():
                            mask = contains_mask
                            break
                    
                if mask.any():
                    row = df[mask].iloc[0]
                    for metric in metrics_to_plot:
                        col_name = f"{metric}_mean"
                        if col_name in row.index:
                            gathered_data[ds_name][metric][meth] = float(row[col_name])
        except Exception as e:
            pass

    # Plotting: 10 datasets in a 2x5 grid with shared Y-axis
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)
    axes = axes.flatten() 

    bar_width = 0.12 # Rộng hơn 1 chút vì 6 cột
    x_indexes = np.arange(len(metrics_to_plot))
    y_ticks = np.arange(0.0, 1.01, 0.1)
    
    # 6 colors for 6 methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, ds in enumerate(dataset_order):
        ax = axes[i]
        
        ds_all_scores = []
        for j, method in enumerate(methods_order):
            scores = [gathered_data[ds][m][method] for m in metrics_to_plot]
            ds_all_scores.extend(scores)
            
            # Offset cho 6 cột
            offset = (j * bar_width) - (bar_width * 2.5)
            ax.bar(x_indexes + offset, scores, 
                   width=bar_width, 
                   label=method, 
                   color=colors[j],
                   edgecolor='black')
            
        ax.set_title(ds.capitalize(), fontweight='bold')
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(metrics_to_plot)
        
        # Chỉ vẽ nhãn trục Y cho các biểu đồ nằm ở cột đầu tiên của mỗi hàng
        if i % 5 == 0:
            ax.set_ylabel('Score (0 - 1.0)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        
        # Cố định thang đo trục Y từ 0 đến 1 
        ax.set_ylim(0, 1.05)
        ax.set_yticks(y_ticks)
        
        # Chỉ giữ lưới ngang để hỗ trợ so sánh chiều cao cột
        ax.grid(axis='y', linestyle='-', alpha=0.3)

        # Bỏ viền trên/phải để biểu đồ thoáng và hiện đại hơn
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Thêm chú thích màu
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    
    # Lưu file theo mốc thời gian
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, f'KB2_10_Datasets_BarChart_{timestamp}.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    plot_kb2_by_dataset()
