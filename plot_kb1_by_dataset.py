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

def plot_kb1_by_dataset(base_dir=None, dataset_order=None, metrics_to_plot=None):
    if base_dir is None:
        base_dir = r"d:\01_Study\Paper\paper_resources\01_src\afw-cil\notebook_modified\Experiment"
        
    if dataset_order is None:
        dataset_order = ['pima', 'haberman', 'vehicle3', 'yeast']
        
    if metrics_to_plot is None:
        metrics_to_plot = ['Gmean', 'SE', 'AUC'] 
        
    methods_order = ['W-SVM', 'AFW-CIL_default', 'GridSearch_AFW', 'PSO-AFW-CIL']
    
    # Store aggregated scores: dataset_name -> {metric: {method: score}}
    gathered_data = {ds: {m: {meth: 0.0 for meth in methods_order} for m in metrics_to_plot} for ds in dataset_order}

    # Extract Data from CSVs
    for ds_name in dataset_order:
        search_pattern = os.path.join(base_dir, f"Test_{ds_name}_*", f"*KB1_summary*.csv")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            continue
        
        latest_file = sorted(matching_files)[-1]
        
        try:
            df = pd.read_csv(latest_file)
            for meth in methods_order:
                clean_meth_name = meth.replace("-AFW-CIL", "")
                mask = df['Name Method'].str.contains(clean_meth_name, case=False, na=False)
                if mask.any():
                    row = df[mask].iloc[0]
                    for metric in metrics_to_plot:
                        col_name = f"{metric}_mean"
                        if col_name in row.index:
                            gathered_data[ds_name][metric][meth] = float(row[col_name])
        except Exception as e:
            pass

    # Plotting for 4 datasets in one row with shared Y-axis
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)
    axes = axes.flatten() # Phẳng hóa mảng axes 2D thành 1D để dễ lặp

    bar_width = 0.1
    x_indexes = np.arange(len(metrics_to_plot))
    y_ticks = np.arange(0.0, 1.01, 0.1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, ds in enumerate(dataset_order):
        ax = axes[i]
        
        ds_all_scores = []
        for j, method in enumerate(methods_order):
            # Lấy list điểm của phương pháp này trên 3 độ đo
            scores = [gathered_data[ds][m][method] for m in metrics_to_plot]
            ds_all_scores.extend(scores)
            
            # Vẽ cột cho thuật toán đó tại vị trí của 3 độ đo
            offset = (j * bar_width) - (bar_width * 1.5)
            ax.bar(x_indexes + offset, scores, 
                   width=bar_width, 
                   label=method, 
                   color=colors[j],
                   edgecolor='black')
            
        ax.set_title(ds.capitalize(), fontweight='bold')
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(metrics_to_plot)
        
        # Chỉ giữ nhãn số trục Y ở biểu đồ đầu tiên để giảm nhiễu
        if i == 0:
            ax.set_ylabel('Score (0 - 1.0)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        
        # Cố định thang đo trục Y từ 0 đến 1 (thêm 0.05 để cột không chạm nóc)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(y_ticks)
        
        # Chỉ giữ lưới ngang để hỗ trợ so sánh chiều cao cột
        ax.grid(axis='y', linestyle='-', alpha=0.3)

        # Bỏ viền trên/phải để biểu đồ thoáng và hiện đại hơn
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Thêm chú thích màu (legend) chia chung cho toàn bộ Figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))

    # Căn chỉnh để các đồ thị không dính nhãn vào nhau, và chừa chỗ cho legend
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    
    # Save the figure with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, f'KB1_10_Datasets_BarChart_{timestamp}.png')
    plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    print(f"Chart saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    plot_kb1_by_dataset()
