"""
Scenario 3 (SE vs. IR): Publication-ready 1x5 line chart.

This script generates a high-quality multi-panel figure for five datasets,
showing Sensitivity (SE) trends as Imbalance Ratio (IR) decreases.

You can replace the dummy values in the `datasets` dictionary with values
loaded from your CSV files later.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def build_scenario3_se_figure(output_path: str = "KB3_SE_Trend.png") -> None:
    """Create and save the Scenario 3 publication-ready figure."""

    # ------------------------------------------------------------------
    # 1) Global style configuration for journal-quality readability.
    # ------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
        }
    )

    # ------------------------------------------------------------------
    # 2) Structured data container (dummy values).
    #    Replace these lists with your real values from CSV files.
    # ------------------------------------------------------------------
    datasets = {
        "glass1": {
            "IR": [0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
            "W-SVM": [0.00, 0.00, 0.00, 0.02, 0.08, 0.08, 0.80],
            "AFW-CIL_default": [0.30, 0.20, 0.17, 0.14, 0.45, 0.20, 0.50],
            "Proposed": [0.52, 0.53, 0.48, 0.37, 0.60, 0.00, 0.50],
        },
        "vehicle1": {
            "IR": [0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
            "W-SVM": [0.35, 0.33, 0.30, 0.28, 0.25, 0.22, 0.18],
            "AFW-CIL_default": [0.42, 0.40, 0.38, 0.36, 0.34, 0.31, 0.27],
            "Proposed": [0.55, 0.53, 0.50, 0.48, 0.45, 0.43, 0.40],
        },
        "transfusion": {
            "IR": [0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
            "W-SVM": [0.26, 0.24, 0.21, 0.19, 0.17, 0.15, 0.12],
            "AFW-CIL_default": [0.38, 0.36, 0.33, 0.30, 0.28, 0.25, 0.23],
            "Proposed": [0.49, 0.47, 0.45, 0.43, 0.40, 0.38, 0.35],
        },
        "abalone": {
            "IR": [0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
            "W-SVM": [0.31, 0.29, 0.26, 0.24, 0.20, 0.16, 0.13],
            "AFW-CIL_default": [0.44, 0.42, 0.39, 0.36, 0.33, 0.29, 0.25],
            "Proposed": [0.57, 0.55, 0.52, 0.50, 0.47, 0.44, 0.41],
        },
        "bankrupt": {
            "IR": [0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02],
            "W-SVM": [0.40, 0.38, 0.35, 0.32, 0.29, 0.25, 0.21],
            "AFW-CIL_default": [0.51, 0.49, 0.47, 0.44, 0.41, 0.38, 0.34],
            "Proposed": [0.63, 0.61, 0.58, 0.56, 0.53, 0.50, 0.47],
        },
    }

    dataset_order = ["glass1", "vehicle1", "transfusion", "abalone", "bankrupt"]

    # ------------------------------------------------------------------
    # 3) Styling specs for the three compared methods.
    # ------------------------------------------------------------------
    method_plot_specs = {
        "W-SVM": {
            "label": "W-SVM",
            "color": "gray",
            "linestyle": "--",
            "marker": "o",
            "linewidth": 1.5,
            "markersize": 6,
        },
        "AFW-CIL_default": {
            "label": "AFW-CIL_default",
            "color": "darkorange",
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 1.5,
            "markersize": 6,
        },
        "Proposed": {
            "label": "BL-SMOTE+PSO-AFW-CIL",
            "color": "steelblue",
            "linestyle": "-",
            "marker": "s",
            "linewidth": 2.0,
            "markersize": 6,
        },
    }

    # ------------------------------------------------------------------
    # 4) Create figure with 1x5 layout and shared Y-axis.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    # ------------------------------------------------------------------
    # 5) Draw each dataset in a dedicated subplot.
    # ------------------------------------------------------------------
    for i, dataset_name in enumerate(dataset_order):
        ax = axes[i]
        data = datasets[dataset_name]

        ir_values = data["IR"]
        n_points = len(ir_values)

        # Basic consistency check so replacement with CSV values is safer.
        for method_name in method_plot_specs:
            if len(data[method_name]) != n_points:
                raise ValueError(
                    f"Length mismatch in '{dataset_name}' for '{method_name}': "
                    f"expected {n_points}, got {len(data[method_name])}."
                )

        # Plot each method using the required visual style.
        for method_name, spec in method_plot_specs.items():
            ax.plot(
                ir_values,
                data[method_name],
                label=spec["label"],
                color=spec["color"],
                linestyle=spec["linestyle"],
                marker=spec["marker"],
                linewidth=spec["linewidth"],
                markersize=spec["markersize"],
            )

        ax.set_title(dataset_name, fontweight="bold")
        ax.set_xlabel("IR")

        # Invert X-axis so IR is shown from high to low (stress-test direction).
        ax.invert_xaxis()

        # Set common Y-axis limits and subtle dashed grid.
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Only first subplot has Y-axis label, as requested.
        if i == 0:
            ax.set_ylabel("Sensitivity (SE)")

    # ------------------------------------------------------------------
    # 6) Add one unified legend at the bottom center (outside subplots).
    # ------------------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    # Global spacing controls to avoid overlap with external legend.
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))

    # High-resolution export for papers.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    build_scenario3_se_figure("KB3_SE_Trend.png")
