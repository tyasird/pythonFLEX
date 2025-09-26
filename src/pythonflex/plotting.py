# Standard library
from itertools import combinations
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import get_cmap
from matplotlib.ticker import NullFormatter, NullLocator

# Completely disable LaTeX and clear all font cache/references
import matplotlib as mpl
import matplotlib.font_manager as fm

# Disable LaTeX rendering completely
mpl.rcParams['text.usetex'] = False


# Reset all font-related parameters to system defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Bitstream Vera Serif', 'serif']
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Bitstream Vera Sans', 'sans-serif']
mpl.rcParams['font.cursive'] = ['Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT', 'Felipa', 'cursive']
mpl.rcParams['font.fantasy'] = ['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans', 'fantasy']
mpl.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Computer Modern Typewriter', 'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace']

# Remove any LaTeX-specific math font settings
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['mathtext.default'] = 'regular'

# Force font manager to rebuild with system fonts only
try:
    fm.fontManager.__init__()
except:
    pass


# Local modules
from .utils import dload  
from .logging_config import log





def plot_precision_recall_curve(line_width=2.0, hide_minor_ticks=True):
    pra = dload("pra")
    config = dload("config")
    plot_config = config["plotting"]

    fig, ax = plt.subplots()
    ax.set_xscale("log")

    # optionally hide minor ticks on the log axis
    if hide_minor_ticks:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())

    cmap = get_cmap()
    num_colors = len(pra) if isinstance(pra, dict) else 1
    colors = [cmap(float(i) / max(num_colors - 1, 1)) for i in range(num_colors)]

    if isinstance(pra, dict):
        for (key, val), color in zip(pra.items(), colors):
            val = val[val.tp > 10]
            ax.plot(val.tp, val.precision, c=color, label=key, linewidth=line_width, alpha=0.9)
    else:
        pra = pra[pra.tp > 10]
        ax.plot(pra.tp, pra.precision, c="black", label="Precision Recall Curve", linewidth=line_width, alpha=0.9)

    ax.set(title="Precision-Recall Performance of Datasets",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 1)

    # Nature style: no grid, open top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_path = Path(config["output_folder"]) / f"precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()
    plt.close(fig)



def plot_percomplex_scatter(n_top=10, sig_color='#B71A2A', nonsig_color='#DBDDDD', label_color='black', border_color='black', border_width=1.0):
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")

    if len(rdict) < 2:
        print("Skipping plot: At least two datasets are required for per-complex scatter plot.")
        return

    column_pairs = list(combinations(rdict.keys(), 2))
    df = pd.DataFrame()

    for i, (key, val) in enumerate(rdict.items()):
        val = val.rename(columns={"auc_score": key})
        if i == 0:
            df = val.copy().drop(columns=["Genes", "Length", "used_genes"], errors="ignore")
        else:
            df = pd.concat([df, val[key]], axis=1)

    for pair in column_pairs:
        extreme_indices_0 = df[pair[0]].sort_values(ascending=False).head(n_top).index
        extreme_indices_1 = df[pair[1]].sort_values(ascending=False).head(n_top).index
        significant_indices = extreme_indices_0.union(extreme_indices_1)

        bg_df  = df.drop(index=significant_indices)
        sig_df = df.loc[significant_indices]

        fig, ax = plt.subplots()

        # Background cloud (filled dots with black borders, not rasterized)
        bg_sizes = (bg_df['n_used_genes'] if 'n_used_genes' in bg_df else pd.Series(1, index=bg_df.index)) * 5
        ax.scatter(
            bg_df[pair[0]], bg_df[pair[1]],
            facecolors=nonsig_color, edgecolors=border_color,
            s=bg_sizes, linewidth=border_width, alpha=1.0,
            zorder=0
        )

        # Significant points (filled dots with black borders)
        sig_sizes = (sig_df['n_used_genes'] if 'n_used_genes' in sig_df else pd.Series(1, index=sig_df.index)) * 8
        ax.scatter(
            sig_df[pair[0]], sig_df[pair[1]],
            facecolors=sig_color, edgecolors=border_color,
            s=sig_sizes, linewidth=border_width, zorder=2
        )

        # Label only significant with adaptive spacing
        coords = sorted(
            [(sig_df.loc[idx, pair[0]], sig_df.loc[idx, pair[1]], idx) for idx in sig_df.index],
            key=lambda c: (-c[1], -c[0])
        )
        
        # Calculate proper parameters for normalized coordinate system (0-1 range)
        max_y = 1.0  # Normalized plots use 0-1 range
        scale_factor = 1.0  # Standard scaling for normalized plots
        min_distance = 0.05  # Appropriate spacing for 0-1 range
        
        adjusted_coords = adjust_text_positions(
            coords, sig_sizes,
            min_distance=min_distance,
            max_y=max_y,
            scale_factor=scale_factor
        )
        
        for x, adj_y, idx in adjusted_coords:
            y = df.loc[idx, pair[1]]
            ax.plot([x, x], [y, adj_y], color=label_color, linewidth=0.6, alpha=0.3, zorder=3)
            ax.text(
                x, adj_y + 0.005,
                df.loc[idx, 'Name'][:15] + '..',
                fontsize=4, ha='left', va='bottom', color=label_color,
                linespacing=1.5, zorder=4,
                #bbox=dict(facecolor="white", alpha=0.65, edgecolor="white", pad=1.2)
            )

        # Diagonal & axes cosmetics
        ax.plot([0, 1], [0, 1], linestyle='-', color='lightgray', alpha=0.4, linewidth=0.5, zorder=1)
        padding = 0.02
        ax.set_xlim(-padding, 1 + padding)
        ax.set_ylim(-padding, 1 + padding)
        ax.set_xlabel(f"{pair[0]} PR-AUC score")
        ax.set_ylabel(f"{pair[1]} PR-AUC score")
        ax.set_title(f"{pair[0]} vs {pair[1]} - Comparison of complex performance")

        # Nature style: no grid, open top/right spines
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if plot_config["save_plot"]:
            output_type = plot_config["output_type"]
            output_path = Path(config["output_folder"]) / f"percomplex_scatter_{pair[0]}_vs_{pair[1]}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()

        plt.close(fig)



def smart_direction_assignment(point_y, y_max, min_safe_distance=20.0):
    """Determine the best direction for label placement based on Y position."""
    lower_threshold = y_max / 3
    upper_threshold = 2 * y_max / 3
    
    if point_y < lower_threshold:
        return "up_only"
    elif point_y > upper_threshold:
        return "prefer_down"
    else:
        return "both_directions"


def group_points_by_y_proximity(coords, y_tolerance=5.0):
    """Group points that have similar Y values (within tolerance)."""
    groups = []
    remaining_coords = coords.copy()
    
    while remaining_coords:
        # Start a new group with the first remaining point
        seed_point = remaining_coords.pop(0)
        current_group = [seed_point]
        seed_y = seed_point[1]
        
        # Find all points within Y tolerance of the seed point
        i = 0
        while i < len(remaining_coords):
            if abs(remaining_coords[i][1] - seed_y) <= y_tolerance:
                current_group.append(remaining_coords.pop(i))
            else:
                i += 1
                
        groups.append(current_group)
    
    return groups


def adjust_text_positions(coords, sizes, min_distance=0.08, max_y=1.0, scale_factor=1.0):
    """Enhanced text positioning with adaptive spacing for dense clusters."""
    adjusted = []
    
    # Fix scaling issues - use data coordinates, not pixel scaling
    if max_y > 10:  # For gene count plots (large Y values)
        text_height = max_y * 0.02  # 2% of Y range
        min_safe_distance = max_y * 0.05  # 5% of Y range  
        y_tolerance = max_y * 0.02  # 2% of Y range for grouping
    else:  # For normalized plots (Y values 0-1)
        text_height = 0.04 * scale_factor
        min_safe_distance = 20 * scale_factor
        y_tolerance = 5 * scale_factor
    
    # Group points by Y proximity
    groups = group_points_by_y_proximity(coords, y_tolerance)
    
    for group in groups:
        group_size = len(group)
        
        # Calculate adaptive spacing based on cluster density
        density_multiplier = calculate_density_multiplier(group_size)
        
        if group_size == 1:
            # Single point - use original logic but with direction awareness
            x, y, idx = group[0]
            direction = smart_direction_assignment(y, max_y, min_safe_distance)
            
            # Use reasonable base offset relative to Y range
            if max_y > 10:  # Gene count plots
                base_offset = max(3, max_y * 0.03)  # 3% of Y range, minimum 3 units
            else:  # Normalized plots  
                base_offset = np.sqrt(sizes.loc[idx]) * 0.04 * scale_factor if idx in sizes else 0.04 * scale_factor
            
            if direction == "up_only" or direction == "both_directions":
                adj_y = y + base_offset
            elif direction == "prefer_down" and y - base_offset > min_safe_distance:
                adj_y = y - base_offset
            else:
                adj_y = y + base_offset
                
            # Ensure within bounds with proper limits
            adj_y = max(min_safe_distance, min(adj_y, max_y - text_height))
            
            # Additional safety check to prevent extreme values
            if adj_y < 0 or adj_y > max_y * 1.2:  # Allow 20% overflow for safety
                adj_y = y + base_offset  # Fallback to simple offset
            
            adjusted.append((x, adj_y, idx))
            
        else:
            # Multiple points with similar Y - use adaptive distribution
            group.sort(key=lambda p: p[0])  # Sort by X coordinate
            
            # Determine available directions for this Y level
            group_y = group[0][1]  # All have similar Y, use first as representative
            direction = smart_direction_assignment(group_y, max_y, min_safe_distance)
            
            # Calculate adaptive spacing and base offset
            adaptive_spacing = calculate_adaptive_spacing(
                group_size, min_distance, text_height, max_y, density_multiplier
            )
            adaptive_base_offset = calculate_adaptive_base_offset(
                group_size, max_y, scale_factor, density_multiplier
            )
            
            for i, (x, y, idx) in enumerate(group):
                if direction == "up_only":
                    # Stack all labels upward with adaptive spacing
                    adj_y = y + adaptive_base_offset + (i * adaptive_spacing)
                    
                elif direction == "prefer_down":
                    # Alternate down and up with adaptive spacing
                    if i % 2 == 0 and y - adaptive_base_offset - (i//2 * adaptive_spacing) > min_safe_distance:
                        # Even indices go down
                        adj_y = y - adaptive_base_offset - (i//2 * adaptive_spacing)
                    else:
                        # Odd indices or insufficient space below - go up
                        up_level = (i//2) if i % 2 == 0 else ((i+1)//2)
                        adj_y = y + adaptive_base_offset + (up_level * adaptive_spacing)
                        
                else:  # both_directions
                    # Alternate up and down with adaptive spacing
                    if i % 2 == 0:
                        # Even indices go up
                        adj_y = y + adaptive_base_offset + (i//2 * adaptive_spacing)
                    else:
                        # Odd indices go down (if safe)
                        potential_down = y - adaptive_base_offset - ((i+1)//2 * adaptive_spacing)
                        if potential_down > min_safe_distance:
                            adj_y = potential_down
                        else:
                            # Not safe to go down, stack upward instead
                            adj_y = y + adaptive_base_offset + (i//2 * adaptive_spacing)
                
                # Final bounds check with stricter limits
                adj_y = max(min_safe_distance, min(adj_y, max_y - text_height))
                
                # Additional safety check to prevent extreme values
                if adj_y < 0 or adj_y > max_y * 1.2:  # Allow 20% overflow for safety
                    adj_y = y + adaptive_base_offset  # Fallback to simple offset
                
                adjusted.append((x, adj_y, idx))
    
    return adjusted


def calculate_density_multiplier(group_size):
    """Calculate multiplier for spacing based on cluster density."""
    if group_size <= 3:
        return 1.0
    elif group_size <= 6:
        return 1.3
    elif group_size <= 10:
        return 1.6
    elif group_size <= 15:
        return 2.0
    elif group_size <= 20:
        return 2.5
    else:  # 20+ points
        return 3.0 + (group_size - 20) * 0.1  # Progressive scaling for very dense clusters


def calculate_adaptive_spacing(group_size, min_distance, text_height, max_y, density_multiplier):
    """Calculate adaptive vertical spacing between labels based on cluster density."""
    base_spacing = max(min_distance, text_height * 1.5)
    
    # Scale spacing based on density and coordinate system
    if max_y > 10:  # Gene count plots
        adaptive_spacing = base_spacing * density_multiplier * (max_y / 100.0)
        # Ensure minimum readable spacing for dense clusters
        adaptive_spacing = max(adaptive_spacing, max_y * 0.03)
    else:  # Normalized plots
        adaptive_spacing = base_spacing * density_multiplier
        # Ensure minimum readable spacing
        adaptive_spacing = max(adaptive_spacing, 0.05)
    
    return adaptive_spacing


def calculate_adaptive_base_offset(group_size, max_y, scale_factor, density_multiplier):
    """Calculate adaptive base offset (connector line height) based on cluster density."""
    if max_y > 10:  # Gene count plots
        base_offset = max(3, max_y * 0.03)
        # Increase connector line height for dense clusters
        adaptive_offset = base_offset * density_multiplier
        # Cap to reasonable maximum
        adaptive_offset = min(adaptive_offset, max_y * 0.15)
    else:  # Normalized plots
        base_offset = 0.04 * scale_factor
        # Increase connector line height for dense clusters
        adaptive_offset = base_offset * density_multiplier
        # Cap to reasonable maximum
        adaptive_offset = min(adaptive_offset, 0.2)
    
    return adaptive_offset


def plot_percomplex_scatter_bysize(n_labels=10, n_top=10, sig_color='#B71A2A', nonsig_color='#DBDDDD', label_color='black', border_color='black', border_width=1.0):
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")

    for key, per_complex in rdict.items():
        sorted_pc = per_complex.sort_values(by="auc_score", ascending=False, na_position="last")
        top_labels, rest = sorted_pc.head(n_labels), sorted_pc.iloc[n_labels:]

        fig, ax = plt.subplots()

        # Background (REST): filled dots with black borders, not rasterized
        ax.scatter(
            rest.auc_score, rest.n_used_genes,
            facecolors=nonsig_color, edgecolors=border_color,
            linewidth=border_width, s=rest.n_used_genes * 10,
            alpha=1.0, label="Other Complexes",
            zorder=0
        )

        # Top N: filled dots with black borders
        ax.scatter(
            top_labels.auc_score, top_labels.n_used_genes,
            facecolors=sig_color, edgecolors=border_color,
            linewidth=border_width, s=top_labels.n_used_genes * 10,
            label=f"Top {n_labels} AUC Scores", alpha=1.0, zorder=2
        )

        # Labels with corrected scaling
        coords = [(row.auc_score, row.n_used_genes, idx) for idx, row in top_labels.iterrows()]
        sizes = top_labels.n_used_genes * 10
        max_y = sorted_pc.n_used_genes.max() + 50
        
        # Fix scaling issue - use reasonable scale factor
        scale_factor = min(max_y / 100.0, 3.0)  # Cap scale factor to prevent extreme positioning
        
        adjusted = adjust_text_positions(
            coords, sizes,
            min_distance=max(5.0, max_y * 0.02),  # Use reasonable spacing relative to Y range
            max_y=max_y, 
            scale_factor=scale_factor
        )
        for x, adj_y, idx in adjusted:
            y = top_labels.loc[idx, "n_used_genes"]
            ax.plot([x, x], [y, adj_y], color=label_color, linewidth=0.5, alpha=0.3, zorder=3)
            ha = 'left' if x < 0.5 else 'right'
            text_x = x + 0.01 if x < 0.5 else x - 0.01
            ax.text(
                text_x, adj_y + (0.005 * scale_factor),
                top_labels.loc[idx, 'Name'][:15] + '..',
                fontsize=4, ha=ha, va='bottom', color=label_color, linespacing=1.5, zorder=4,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="white", pad=1.5)
            )

        # Set y-axis to show integer values only
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("PR-AUC score")
        ax.set_ylabel("Number of genes in the complex")
        ax.set_title(f"{key} - Complex performance: PR-AUC score vs complex size")

        # No ruler + open spines
        ax.grid(visible=False, which='both', axis='both')
        ax.set_xlim(0, 1.0); ax.set_ylim(0, max_y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.subplots_adjust(right=0.8)
        plt.tight_layout()

        if plot_config["save_plot"]:
            output_type = plot_config["output_type"]
            output_path = Path(config["output_folder"]) / f"percomplex_scatter_by_complexsize_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()
        plt.close(fig)



def plot_complex_contributions(
    min_pairs=10,
    min_precision_cutoff=0.5,
    num_complex_to_show=10,
    y_lim=None,
    fig_title=None,
    fig_labs=['Fraction of TP', 'Precision'],
    legend_rows=3,   # <— NEW: rows for legend layout (try 3 or 4)
):
    config = dload("config")
    plot_config = config["plotting"]
    plot_data_dict = dload("complex_contributions")

    for key, plot_data in plot_data_dict.items():
        s = plot_data.set_index('Name').sum()
        find_last_precision = s[s > min_pairs].index[-1]
        last_prec_value = float(find_last_precision.split('_')[1])

        plot_data = plot_data.drop_duplicates(subset='Name')
        cont_stepwise_anno = plot_data['Name']
        cont_stepwise_mat = plot_data.drop(columns=['Name'])
        tmp_TP = cont_stepwise_mat.sum(axis=0)
        Precision_ind = (tmp_TP >= min_pairs)
        cont_stepwise_mat = cont_stepwise_mat.loc[:, Precision_ind]
        tmp = cont_stepwise_mat.columns
        y = np.array([float(col.split('_')[1]) if isinstance(col, str) and '_' in col else col for col in tmp])
        x = cont_stepwise_mat.sum(axis=0)
        mx, nx = cont_stepwise_mat.shape[0], cont_stepwise_mat.shape[1]
        tmp = np.tile(x, (mx, 1))
        x = cont_stepwise_mat.values / tmp
        x_df = pd.DataFrame(x, index=cont_stepwise_anno, columns=cont_stepwise_mat.columns)
        ind_for_mean = y >= (last_prec_value - min_precision_cutoff)
        if sum(ind_for_mean) == 0:
            log.info("No values above 'min.precision.cutoff'"); return False
        if sum(ind_for_mean) == 1:
            log.info("Only one value above 'min.precision.cutoff'"); return False

        a = x_df.loc[:, ind_for_mean].mean(axis=1).sort_values()[-num_complex_to_show:]
        subset = x_df.loc[a.index, :]

        cmap = plt.get_cmap()
        colors = cmap(np.linspace(0, 1, num_complex_to_show))
        colors = np.vstack(([0.5, 0.5, 0.5, 1.0], colors))  # 'others' + top K
        others = pd.DataFrame(1 - subset.sum(axis=0), columns=['others']).T
        merged = pd.concat([others, subset], ignore_index=False)
        X = merged.to_numpy()
        x1 = np.zeros_like(X); x2 = np.zeros_like(X)
        for i in range(X.shape[0]):
            if i == 0:
                x2[i, :] = X[0, :]
            elif i == 1:
                x1[i, :] = X[0, :]
            else:
                x1[i, :] = X[:i, :].sum(axis=0)
            if i > 0:
                x2[i, :] = X[:i + 1, :].sum(axis=0)

        padding = 0.02
        lower = max(0, min(y) - padding)
        upper = last_prec_value + padding
        y_lim = (lower, upper)

        # Give legend a bit more room
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1.8]})
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(*y_lim)
        ax[0].set_xlabel(fig_labs[0])
        ax[0].set_ylabel(fig_labs[1])
        ax[0].set_title(fig_title if fig_title else f"{key} - Contribution of complexes")
        for i in range(X.shape[0]):
            ax[0].fill_betweenx(y, x1[i, :], x2[i, :], color=colors[i], edgecolor='white')

        # Legend: multi-row, constrained to width
        def _short(s, n=14): return (s[:n-1] + '…') if len(s) > n else s
        labels  = [_short(lbl) for lbl in merged.index]
        handles = [patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax[1].axis('off')
        n_items = len(handles)
        ncols   = int(np.ceil(n_items / max(1, legend_rows)))  # spread across rows
        ax[1].legend(
            handles=handles,
            loc='center',
            ncol=ncols,
            frameon=False,
            title="Complexes",
            fontsize=6, title_fontsize=6,
            handlelength=0.9, handletextpad=0.25,
            borderaxespad=0.0,
            labelspacing=0.25, columnspacing=0.6,
            mode='expand'
        )

        plt.tight_layout()

        if plot_config["save_plot"]:
            output_type  = plot_config["output_type"]
            output_folder= Path(config["output_folder"])
            output_path  = output_folder / f"complex_contributions_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        if plot_config.get("show_plot", True):
            plt.show()
        plt.close(fig)



def plot_significant_complexes():
    config = dload("config")
    plot_config = config["plotting"]
    pra_percomplex = dload("pra_percomplex")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    datasets = list(pra_percomplex.keys())
    num_datasets = len(datasets)

    df = pd.DataFrame(index=thresholds)
    for key, complex_data in pra_percomplex.items():
        df[key] = [complex_data.query(f'auc_score >= {t}').shape[0] for t in thresholds]

    fig, ax = plt.subplots()

    cmap = plt.get_cmap()
    colors = [cmap(i / (num_datasets + 1)) for i in range(1, num_datasets + 1)]

    bar_width = 0.8 / num_datasets
    for i, dataset in enumerate(datasets):
        x = np.arange(len(thresholds)) + i * bar_width
        ax.bar(x, df[dataset], width=bar_width, color=colors[i], edgecolor='black', label=dataset)

    ax.set_xticks(np.arange(len(thresholds)) + (num_datasets - 1) * bar_width / 2)
    ax.set_xticklabels([str(t) for t in thresholds], rotation=0, ha='center')

    ax.set_title("Number of significant complexes above PR-AUC thresholds")
    ax.set_xlabel("PR-AUC score thresholds")
    ax.set_ylabel("Number of complexes")

    # Nature style: no grid; open top/right spines
    ax.grid(False)
    for spine in ('right', 'top'):
        ax.spines[spine].set_visible(False)

    ax.legend(loc='upper right', frameon=False)
    plt.tight_layout()

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_folder = Path(config["output_folder"])
        output_path = output_folder / f"number_of_significant_complexes.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()

    plt.close(fig)
    return df



def plot_auc_scores():
    config = dload("config")
    plot_config = config["plotting"]
    pra_dict = dload("pr_auc")

    sorted_items = sorted(pra_dict.items(), key=lambda x: x[1], reverse=True)
    datasets = [k for k, _ in sorted_items]
    auc_scores = [v for _, v in sorted_items]

    fig, ax = plt.subplots()

    cmap = plt.get_cmap()
    num_datasets = len(datasets)
    colors = [cmap(i / (num_datasets + 1)) for i in range(1, num_datasets + 1)]

    ax.bar(datasets, auc_scores, color=colors, edgecolor="black")

    ax.set_ylim(0, max(auc_scores) + 0.01)
    ax.set_title("AUC scores for the datasets")
    ax.set_ylabel("AUC score")
    plt.xticks(rotation=45, ha="right")

    # Hard-disable any grid/ruler
    ax.grid(visible=False, which='both', axis='both')
    ax.set_axisbelow(False)  # make sure nothing faint is drawn beneath
    # Open spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plot_config["save_plot"]:
        output_type = plot_config["output_type"]
        output_folder = Path(config["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"prauc_scores.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    if plot_config.get("show_plot", True):
        plt.show()

    plt.close(fig)
    return pra_dict
