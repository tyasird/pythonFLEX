# Standard library imports
import gc
import os
import re
import shutil
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

# Third-party imports
from art import tprint
from bitarray import bitarray
from joblib import Parallel, delayed, dump, load
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

# Local/application-specific imports
from .logging_config import log
from .preprocessing import filter_matrix_by_genes
from .utils import dsave, dload, _sanitize



def deep_update(source, overrides):
    """Recursively update the source dict with the overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source



def initialize(config={}):

    default_config = {
        "min_genes_in_complex": 3,
        "min_genes_per_complex_analysis": 3,
        "output_folder": "output",
        "gold_standard": "CORUM",
        "color_map": "RdYlBu",
        "jaccard": True,
        "plotting": {
            "save_plot": True,
            "show_plot": True,
            "output_type": "png",
        },
        "preprocessing": {
            "normalize": False,
            "fill_na": False,
            "drop_na": False,
        },
        "corr_function": "numpy",
        "logging": {  # Added: Default logging config
            "visible_levels": ["DONE"]  # if needed #, "PROGRESS", "STARTED", "INFO"
        }
    }
    
    # Early merge to get user-overridden config (including logging.visible_levels)
    if config is not None:
        config = deep_update(default_config, config)
    else:
        config = default_config
    
    # Extract visible_levels from the merged config and set logging visibility immediately (before any logs)
    visible_levels = config.get("logging", {}).get("visible_levels", ["DONE"])
    log.set_visible_levels(visible_levels)

    log.info("******************************************************************")
    log.info("🧬 pyFLEX: Systematic CRISPR screen benchmarking framework")
    log.info("******************************************************************")
    log.started("Initialization")

    # Check and remove .tmp folder if it exists (clean slate to avoid overriding old results)
    tmp_folder = ".tmp"
    if os.path.exists(tmp_folder):
        log.info(f"Removing existing '{tmp_folder}' folder for a clean start.")
        shutil.rmtree(tmp_folder)
        log.done(f"'{tmp_folder}' folder removed successfully.")

    log.progress("Saving configuration settings.")   
        
    dsave(config, "config")
    update_matploblib_config(config)
    output_folder = config.get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)
    log.progress(f"Output folder '{output_folder}' ensured to exist.")
    log.done("Initialization completed. ")
    tprint("pyFLEX",font="standard")



def update_matploblib_config(config={}):
    log.progress("Updating matplotlib settings.")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",        # ← change if you prefer Arial, etc.
        "mathtext.fontset": "dejavusans",
        'font.size': 7,                # General font size
        'axes.titlesize': 10,          # Title size
        'axes.labelsize': 7,           # Axis labels (xlabel/ylabel)
        'legend.fontsize': 7,          # Legend text
        'xtick.labelsize': 6,          # X-axis tick labels
        'ytick.labelsize': 6,          # Y-axis tick labels
        'lines.linewidth': 1.5,        # Line width for plots
        'figure.dpi': 300,             # Figure resolution
        'figure.figsize': (8, 6),      # Default figure size
        'grid.linestyle': '--',        # Grid line style
        'grid.linewidth': 0.5,         # Grid line width
        'grid.alpha': 0.2,             # Grid transparency
        'axes.spines.right': False,    # Hide right spine
        'axes.spines.top': False,      # Hide top spine
        'image.cmap': config['color_map'],        # Default colormap
        'axes.edgecolor': 'black',                # Axis edge color
        'axes.facecolor': 'none',                 # Transparent axes background
        'text.usetex': False                # Ensure LaTeX is off
    })
    log.done("Matplotlib settings updated.")





def pra(dataset_name, matrix, is_corr=False):
    log.info(f"******************** {dataset_name} ********************")
    log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")
    config = dload("config")

    terms_data = dload("common", "terms")
    if terms_data is None or not isinstance(terms_data, pd.DataFrame):
        raise ValueError("Expected 'terms' to be a DataFrame, but got None or invalid type.")
    terms = terms_data
    genes_present = dload("common", "genes_present_in_terms")
    sorting = dload("input", "sorting")
    sort_order = sorting.get(dataset_name, "high")

    if not is_corr:
        matrix = perform_corr(matrix, config.get("corr_function"))
        
    matrix = filter_matrix_by_genes(matrix, genes_present)

    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")
    df = quick_sort(df, ascending=(sort_order == "low"))

    log.started("Building gene-to-pair indices")
    gold_pair_to_complex = _build_gold_pair_to_complex(terms)  
    log.done("Gene-to-pair indices built.")
    
    log.started("Precomputing complex IDs")
    df = _precompute_complex_ids(df, gold_pair_to_complex)
    log.done("Complex IDs precomputed.")

    df["prediction"] = df["complex_ids"].astype(bool).astype(int)
    df["complex_id"] = df["complex_ids"].apply(
        lambda s: list(map(int, s.split(";"))) if s else []
    )

    if df["prediction"].sum() == 0:
        log.info("No true positives found in dataset.")
        pr_auc = np.nan
    else:
        tp = df["prediction"].cumsum()
        df["tp"] = tp
        precision = tp / (np.arange(len(df)) + 1)
        recall = tp / tp.iloc[-1]
        pr_auc = metrics.auc(recall, precision)
        df["precision"] = precision
        df["recall"] = recall

    log.info(f"PR-AUC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
    dsave(df, "pra", dataset_name)
    dsave(pr_auc, "pr_auc", dataset_name)
    log.done(f"Global PRA completed for {dataset_name}")
    return df, pr_auc







# --------------------------------------------------------------------------
# helper functions for PRA per-complex analysis
# --------------------------------------------------------------------------

def _build_gene_to_pair_indices(pairwise_df):
    indices = pairwise_df.index.values
    genes = pd.concat([pairwise_df['gene1'], pairwise_df['gene2']], ignore_index=True)
    stacked_indices = np.concatenate([indices, indices])
    idx_series = pd.Series(stacked_indices, index=range(len(genes)))
    gene_to_pair_indices = defaultdict(list)
    for gene, group in idx_series.groupby(genes):
        gene_to_pair_indices[gene] = group.values.tolist() 
    return gene_to_pair_indices


def _build_gold_pair_to_complex(terms):
    pair_map = defaultdict(set)
    for comp_id, genes in zip(terms.index, terms['used_genes']):
        genes = list(genes)
        if len(genes) < 2: continue
        for i in range(len(genes)):
            for j in range(i+1, len(genes)):
                g1, g2 = sorted([genes[i], genes[j]])
                pair_map[(g1, g2)].add(comp_id)
    return pair_map


def _precompute_complex_ids(pairwise_df, gold_pair_to_complex):
    if not gold_pair_to_complex:
        pairwise_df['complex_ids'] = ''
        return pairwise_df
    
    # Precompute pairs as tuples
    g1 = pairwise_df['gene1']
    g2 = pairwise_df['gene2']
    pairs = [tuple(sorted((a, b))) for a, b in zip(g1, g2)]
    pairwise_df['complex_ids'] = [
        ';'.join(map(str, sorted(gold_pair_to_complex[p]))) 
        if p in gold_pair_to_complex else '' 
        for p in pairs
    ]
    return pairwise_df



def _dump_pairwise_memmap(df: pd.DataFrame, tag: str) -> Path:
    tmp_dir = Path(os.path.join(".tmp", "mmap"))  # Use .tmp/mmap/ for organization
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Create if it doesn't exist
    path = tmp_dir / f".pairwise_{_sanitize(tag)}.pkl"          
    dump(df, path, compress=0)  
    return path 




def _init_worker(memmap_path, gene_to_pair_indices):
    global PAIRWISE_DF, GENE2IDX
    PAIRWISE_DF = load(memmap_path)        
    GENE2IDX    = gene_to_pair_indices                   



def delete_memmap(memmap_path, log, wait_seconds=0.1):

    gc.collect()
    time.sleep(wait_seconds)

    try:
        os.remove(memmap_path)
        log.info(f"Cleaned up temporary memmap file: {memmap_path}")
    except OSError as e:
        log.warning(f"Original error: {e}")



# --------------------------------------------------------------------------
# Process each chunk of terms
# --------------------------------------------------------------------------
def _process_chunk(chunk_terms, min_genes):
    pairwise_df = PAIRWISE_DF
    gene_to_pair_indices = GENE2IDX
    local_auc_scores = {}

    for idx, row in chunk_terms.iterrows():
        gene_set = set(row.used_genes)
        if len(gene_set) < min_genes:
            continue

        candidate_indices = bitarray(len(pairwise_df))
        for g in gene_set:
            if g in gene_to_pair_indices:
                candidate_indices[gene_to_pair_indices[g]] = True
        if not candidate_indices.any():
            continue

        selected = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
        sub_df   = pairwise_df.iloc[selected]

        complex_id = str(idx)
        pattern    = r'(?:^|;)' + re.escape(complex_id) + r'(?:;|$)'
        true_label = sub_df["complex_ids"].str.contains(pattern, regex=True).astype(int)
        mask       = (sub_df["complex_ids"] == "") | (true_label == 1)
        preds      = true_label[mask]

        if preds.sum() == 0:
            continue

        tp_cum   = preds.cumsum()
        precision = tp_cum / (np.arange(len(preds)) + 1)
        recall    = tp_cum / tp_cum.iloc[-1]
        if len(recall) >= 2 and recall.iloc[-1] != 0:
            local_auc_scores[idx] = metrics.auc(recall, precision)

    return local_auc_scores



def pra_percomplex(dataset_name, matrix, is_corr=False, chunk_size=200):
    log.started(f"*** Per-complex PRA started - {dataset_name} ***")
    config = dload("config")
    terms = dload("common", "terms")
    genes_present = dload("common", "genes_present_in_terms")
    sorting = dload("input", "sorting")
    sort_order = sorting.get(dataset_name, "highdor")
    if not is_corr:
        matrix = perform_corr(matrix, config.get("corr_function"))
    matrix = filter_matrix_by_genes(matrix, genes_present)
    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")
    df = quick_sort(df, ascending=(sort_order == "low"))
    pairwise_df = df.copy()
    pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
    pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")
    
    # Use helper functions for precomputations
    log.started("Building gene-to-pair indices")
    gene_to_pair_indices = _build_gene_to_pair_indices(pairwise_df)
    log.done("Building gene-to-pair indices") 
    
    log.started("Building gold pair to complex mapping")
    gold_pair_to_complex = _build_gold_pair_to_complex(terms)  # Now serial
    log.done("Building gold pair to complex mapping") 
    
    log.started("Precomputing complex IDs")
    pairwise_df = _precompute_complex_ids(pairwise_df, gold_pair_to_complex)
    log.done("Precomputing complex IDs")  # 

    log.info('Dumping pairwise_df to memmap')
    memmap_path = _dump_pairwise_memmap(pairwise_df, dataset_name)
    log.done('Dumping pairwise_df to memmap')

    # choose smaller chunks now that pickling cost is gone
    chunks = [terms.iloc[i:i+chunk_size] for i in range(0, len(terms), chunk_size)]
    min_genes = config["min_genes_per_complex_analysis"]

    # Initialize results variable
    results = None
    
    try:
        # Simplified parallel execution without progress callback interference
        log.started("Processing chunks in parallel")
        with tqdm(total=len(chunks), desc="Per-complex PRA") as pbar:
            results = Parallel(
                n_jobs=8,
                temp_folder=os.path.dirname(memmap_path),     
                max_nbytes=None,                              
                mmap_mode="r",
                initializer=_init_worker,
                initargs=(memmap_path, gene_to_pair_indices),
                verbose=0  # Reduce joblib verbosity
            )(delayed(_process_chunk)(chunk, min_genes) for chunk in chunks)
            
            # Update progress bar once all tasks are complete
            pbar.update(len(chunks))
        
        log.done("Processing chunks in parallel")
        
    except Exception as e:
        log.error(f"Error during parallel processing: {e}")
        # Still try to clean up the memmap file
        try:
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
                log.info(f"Cleaned up temporary memmap file after error: {memmap_path}")
        except OSError as cleanup_error:
            log.warning(f"Failed to remove memmap file after error {memmap_path}: {cleanup_error}")
        raise  # Re-raise the original exception
    
    finally:
        # Ensure cleanup happens regardless of success or failure
        try:
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
                log.info(f"Cleaned up temporary memmap file: {memmap_path}")
        except OSError as e:
            log.warning(f"Failed to remove memmap file {memmap_path}: {e}")

    # Merge results with error handling
    auc_scores = {}
    if results:
        for res in results:
            if isinstance(res, dict):
                auc_scores.update(res)
            elif isinstance(res, tuple) and res[0] is None:
                log.error(res[1])  # Log the error message from the chunk
            else:
                log.error(f"Ignoring unexpected chunk result: {res}")
    
    # Add the computed AUC scores to the terms DataFrame.
    terms["auc_score"] = pd.Series(auc_scores)
    terms.drop(columns=["hash"], inplace=True)
    dsave(terms, "pra_percomplex", dataset_name)
    log.done(f"Per-complex PRA completed.")
    return terms





def complex_contributions(name):
    log.info(f"Computing complex contributions (Greedy) for dataset: {name}")
    pra = dload("pra", name)
    terms = dload("common", "terms")
    
    # Ensure pra is sorted by score descending (matches R's order by predicted descending)
    pra = pra.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # Compute cumulative TP and precision (matches R's TP.count = cumsum(true), Precision = TP / (1:n))
    pra['cumTP'] = pra['prediction'].cumsum()
    pra['rank'] = pra.index + 1
    pra['precision'] = pra['cumTP'] / pra['rank']
    
    # R-style precision thresholds (matches c( min, seq(0.1, max, 0.025) ) rounded)
    prec_min = pra['precision'].min()
    prec_max = pra['precision'].max()
    precision_cutoffs = [round(prec_min, 3)]
    cutoffs_range = np.arange(0.1, prec_max + 0.001, 0.025)
    precision_cutoffs += [round(t, 3) for t in cutoffs_range if t > prec_min]
    thresholds = sorted(set(precision_cutoffs))  # Ensure unique and sorted
    
    # Precompute positives for faster access
    pos_mask = pra['prediction'] == 1
    positives = pra[pos_mask].reset_index(drop=True)
    
    # Compute global unique ordered IDs for initial tie-breaking (appearance order from all positives)
    global_row_to_cids = []
    for ids in positives['complex_id']:
        if isinstance(ids, str):
            cleaned = [str(int(float(i.strip()))) for i in ids.split(';') if i.strip()]
        else:
            cleaned = [str(int(i)) for i in ids if pd.notnull(i)]
        global_row_to_cids.append(cleaned)
    all_global_ids = [cid for cids in global_row_to_cids for cid in cids]
    global_unique_ordered = list(OrderedDict.fromkeys(all_global_ids))
    
    results = {}
    valid_thresholds = []  # Track valid like R's ind.valid.precision
    
    # Progress bar for the main loop (thresholds)
    with tqdm(total=len(thresholds), desc="Processing thresholds", unit="thresh") as pbar:
        for thresh_idx, t in enumerate(thresholds):
            # Check if valid (matches R's ind.valid.precision)
            if not (prec_min <= t <= prec_max):
                pbar.update(1)
                continue
            valid_thresholds.append(thresh_idx)  # Track for later sorting
            
            # Find rightmost k where precision >= t (matches R's cand.ind[length(cand.ind)])
            cand_mask = pra['precision'] >= t
            if not cand_mask.any():
                pbar.update(1)
                continue
            k = pra.index[cand_mask].max()
            tp_target = pra.loc[k, 'cumTP']
            if tp_target <= 0:
                pbar.update(1)
                continue
            
            # Find first ind where cumTP == tp_target (matches R's tmp.ind[1])
            matching_inds = pra[pra['cumTP'] == tp_target].index
            if matching_inds.empty:
                pbar.update(1)
                continue
            ind = matching_inds.min()  # First (smallest) like R
            
            # Get top (ind+1) rows, filter to prediction==1 and non-null complex_id
            tmp = pra.iloc[0:ind + 1]
            tmp = tmp[(tmp['prediction'] == 1) & tmp['complex_id'].notnull()].reset_index(drop=True)
            if tmp.empty:
                pbar.update(1)
                continue
            
            # Build row_to_cids as list of lists (str for consistency, matches R strsplit)
            row_to_cids = []
            for ids in tmp['complex_id']:
                if isinstance(ids, str):
                    cleaned = [str(int(float(i.strip()))) for i in ids.split(';') if i.strip()]
                else:
                    cleaned = [str(int(i)) for i in ids if pd.notnull(i)]
                row_to_cids.append(cleaned)
            
            N = len(row_to_cids)
            cid_to_rows = defaultdict(list)
            for row_idx in range(N):
                for cid in row_to_cids[row_idx]:
                    cid_to_rows[cid].append(row_idx)
            
            current_size = {cid: len(lst) for cid, lst in cid_to_rows.items()}
            covered = np.zeros(N, dtype=bool)
            remaining_rows = list(range(N))  # Track remaining for tie-breaking
            final_contrib = {}
            is_first = True  # Flag for initial greedy step
            
            while current_size:
                if not current_size:
                    break
                max_contrib = max(current_size.values())
                candidates = [cid for cid, cnt in current_size.items() if cnt == max_contrib]
                
                if len(candidates) == 1:
                    top_cid = candidates[0]
                else:
                    if is_first:
                        # Initial tie-break: first in global appearance order (matches R's global matrix row order)
                        positions = {cid: global_unique_ordered.index(cid) for cid in candidates if cid in global_unique_ordered}
                        top_cid = min(positions, key=positions.get)
                    else:
                        # Subsequent: first in local remaining appearance order
                        all_ids = [cid for ri in remaining_rows for cid in row_to_cids[ri]]
                        unique_ordered = list(OrderedDict.fromkeys(all_ids))
                        positions = {cid: unique_ordered.index(cid) for cid in candidates if cid in unique_ordered}
                        top_cid = min(positions, key=positions.get)  # Earliest appearance
                
                contrib = current_size[top_cid]
                if contrib <= 0:
                    current_size.pop(top_cid, None)
                    continue
                
                # Cover the remaining rows for top_cid
                for row_idx in cid_to_rows[top_cid]:
                    if not covered[row_idx]:
                        covered[row_idx] = True
                        for cid in row_to_cids[row_idx]:
                            if cid in current_size:
                                current_size[cid] -= 1
                                if current_size[cid] <= 0:
                                    current_size.pop(cid, None)
                
                # Update remaining_rows (remove covered)
                remaining_rows = [ri for ri in remaining_rows if not covered[ri]]
                
                final_contrib[top_cid] = contrib
                is_first = False  # Only first time is special
            
            # Store for this threshold
            for cid, count in final_contrib.items():
                if cid not in results:
                    results[cid] = [0] * len(thresholds)
                results[cid][thresh_idx] = count
            
            pbar.update(1)  # Update progress after processing threshold
    
    # Build result DataFrame (index=cid as str)
    r = pd.DataFrame(results, index=thresholds).T
    r.index = r.index.astype(str)
    
    # Filter to non-zero first (matches R's nonzero.cont.ind)
    r = r[r.sum(axis=1) > 0]
    
    # Intersect with terms IDs, preserving terms order 
    gold_ids = set(r.index)
    common_ids = [str(id) for id in terms.index if str(id) in gold_ids]
    r = r.loc[common_ids]
    
    # Map Names and insert as first column
    t = pd.Series(terms['Name'].values, index=terms.index.astype(str))
    r.insert(0, 'Name', r.index.map(t))
    
    # Set all column names: Name + Precision_*
    precision_cols = [f"Precision_{t}" for t in thresholds]
    r.columns = ['Name'] + precision_cols
    
    # Sort by the last valid precision column descending, stable sort (matches R's stable order)
    if valid_thresholds:
        last_valid_col = f"Precision_{thresholds[valid_thresholds[-1]]}"
        r = r.sort_values(by=last_valid_col, ascending=False, kind='stable')
    
    # De-duplicate by Name, keeping first (matches R's !duplicated(Name) after function)
    r = r[~r['Name'].duplicated(keep='first')].reset_index(drop=True)
    
    dsave(r, "complex_contributions", name)
    log.info(f"Complex contribution (Greedy) completed for dataset: {name}")
    return r




# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def perform_corr(df, corr_func):
    if corr_func not in {"numpy", "pandas","numba"}:
        raise ValueError("corr_func must be 'numpy' or 'pandas'")

    log.started(f"Performing correlation using '{corr_func}' method.")
    
    if corr_func == "numpy":
        M    = np.ma.masked_invalid(df.values)
        corr = np.ma.corrcoef(M)
        arr  = corr.filled(np.nan)
        df_corr = pd.DataFrame(arr, index=df.index, columns=df.index)
        np.fill_diagonal(df_corr.values, np.nan)
        log.done("Correlation.")
        return df_corr
    
    elif corr_func == "numba":
        corr = fast_corr(df)
        np.fill_diagonal(corr.values, np.nan)
        log.done("Correlation using Numba.")
        return corr
    
    else:
        # Compute correlations and modify diagonal in-place
        corr = df.T.corr()
        np.fill_diagonal(corr.values, np.nan)
        return corr



def fast_corr(df):
    @njit(parallel=True)
    def compute_corr(data):
        m, n = data.shape
        corr = np.full((n, n), np.nan, dtype=np.float64)
        # Compute off-diagonal (upper triangle, parallel over i)
        for i in prange(n):
            for j in range(i + 1, n):
                sum_x = 0.0
                sum_y = 0.0
                sum_xx = 0.0
                sum_yy = 0.0
                sum_xy = 0.0
                count = 0
                for k in range(m):
                    x = data[k, i]
                    y = data[k, j]
                    if not np.isnan(x) and not np.isnan(y):
                        sum_x += x
                        sum_y += y
                        sum_xx += x * x
                        sum_yy += y * y
                        sum_xy += x * y
                        count += 1
                if count >= 2:
                    # Sample variance/covariance (div by count-1)
                    var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
                    var_y = (sum_yy - (sum_y ** 2) / count) / (count - 1)
                    cov = (sum_xy - (sum_x * sum_y) / count) / (count - 1)
                    denom = np.sqrt(var_x * var_y)
                    if denom > 0:  # Avoid div-by-zero (e.g., constant cols -> nan)
                        r = cov / denom
                    else:
                        r = np.nan
                else:
                    r = np.nan
                corr[i, j] = r
                corr[j, i] = r  # Symmetric
        # Compute diagonal in parallel
        for i in prange(n):
            sum_x = 0.0
            sum_xx = 0.0
            count = 0
            for k in range(m):
                x = data[k, i]
                if not np.isnan(x):
                    sum_x += x
                    sum_xx += x * x
                    count += 1
            if count >= 2:
                var_x = (sum_xx - (sum_x ** 2) / count) / (count - 1)
                if var_x > 0:
                    corr[i, i] = 1.0
                else:
                    corr[i, i] = np.nan  # Constant column
            else:
                corr[i, i] = np.nan
        return corr
    
    df_numeric = df.select_dtypes(include=np.number)
    data = df_numeric.to_numpy().T
    corr_matrix = compute_corr(data)
    corr_df = pd.DataFrame(corr_matrix, index=df_numeric.index, columns=df_numeric.index)
    return corr_df



def is_symmetric(df):
    return np.allclose(df, df.T, equal_nan=True)


def binary(corr):
    log.started("Converting correlation matrix to pair-wise format.")
    if is_symmetric(corr):
        corr = convert_full_to_half_matrix(corr)
    
    stack = corr.stack().rename_axis(index=['gene1', 'gene2']).\
            reset_index().rename(columns={0: 'score'})
    if has_mirror_of_first_pair(stack):
        log.info("Mirror pairs detected. Dropping them to ensure unique gene pairs.")
        stack = drop_mirror_pairs(stack)
    log.done("Pair-wise conversion.")
    return stack


def has_mirror_of_first_pair(df):
    g1, g2 = df.iloc[0]['gene1'], df.iloc[0]['gene2']
    mirror_exists = ((df['gene1'] == g2) & (df['gene2'] == g1)).iloc[1:].any()
    return mirror_exists


def convert_full_to_half_matrix(df):
    if not is_symmetric(df):
        raise ValueError("Matrix must be symmetric to convert to half matrix.")

    log.started("Converting full correlation matrix to upper triangle (half-matrix) format.")
    arr = df.values.copy()
    arr[np.tril_indices_from(arr)] = np.nan  # zero-based lower triangle + diagonal → NaN
    log.done("Matrix conversion.")
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def drop_mirror_pairs(df):
    log.started("Dropping mirror pairs to ensure unique gene pairs (Optimized).")
    gene_pairs = np.sort(df[["gene1", "gene2"]].to_numpy(), axis=1)
    df.loc[:, ["gene1", "gene2"]] = gene_pairs
    df = df.loc[~df.duplicated(subset=["gene1", "gene2"], keep="first")]
    log.done("Mirror pairs are dropped.")
    return df


def quick_sort(df, ascending=False):
    log.started(f"Pair-wise matrix is sorting based on the 'score' column: ascending:{ascending}")
    order = 1 if ascending else -1
    sorted_df = df.iloc[np.argsort(order * df["score"].values)].reset_index(drop=True)
    log.done("Pair-wise matrix sorting.")
    return sorted_df




def save_results_to_csv(categories = ["complex_contributions", "pr_auc", "pra_percomplex"]):

    config = dload("config")  # Load config to get output folder
    output_folder = Path(config.get("output_folder", "output"))
    output_folder = output_folder / "csv"  # Create a subfolder for results
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists
      
    for category in categories:
        data = dload(category)  # Load the data for this category
        if data is None:
            log.warning(f"No data found for category '{category}'. Skipping save.")
            continue
        
        if category == "pr_auc" and isinstance(data, dict):
            # Special handling: Convert dict to DataFrame (assuming keys are indices, values are data)
            # If values are scalars, create a simple DF with 'Dataset' and 'AUC' columns
            try:
                df = pd.DataFrame.from_dict(data, orient='index', columns=['AUC'])
                df.index.name = 'Dataset'
                txt_path = output_folder / f"{category}.txt"
                df.to_csv(txt_path, sep='\t', index=True)  # Save as tab-delimited TXT
                log.info(f"Saved '{category}' as tabular TXT to {txt_path}")
            except Exception as e:
                log.warning(f"Failed to convert and save '{category}' as TXT: {e}")
            continue  # Skip to next category after handling pr_auc
        
        if isinstance(data, dict):
            # If it's a dict of datasets, save each as a separate CSV
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    csv_path = output_folder / f"{category}_{key}.csv"
                    df.to_csv(csv_path, index=False)
                    log.info(f"Saved '{category}_{key}' to {csv_path}")
                else:
                    log.warning(f"Skipping non-DataFrame item '{key}' in '{category}'.")
        elif isinstance(data, pd.DataFrame):
            # If it's a single DataFrame, save it directly
            csv_path = output_folder / f"{category}.csv"
            data.to_csv(csv_path, index=False)
            log.info(f"Saved '{category}' to {csv_path}")
        else:
            log.warning(f"Unsupported data type for '{category}'. Expected DataFrame or dict of DataFrames. Skipping.")

    log.done("Results saved to CSV files in the output folder.")






### OLD FUNCTIONS


# new but withoutparallel

# def pra_percomplex(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started - {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")
#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     pairwise_df = df.copy()
#     pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
#     pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")
    
#     # Precompute a mapping from each gene to the row indices in the pairwise DataFrame where it appears.
#     gene_to_pair_indices = {}
#     for i, (gene_a, gene_b) in enumerate(zip(pairwise_df["gene1"], pairwise_df["gene2"])):
#         gene_to_pair_indices.setdefault(gene_a, []).append(i)
#         gene_to_pair_indices.setdefault(gene_b, []).append(i)
#     log.done
    
#     # Build gold_pair_to_complex using sets for efficiency
#     gold_pair_to_complex = defaultdict(set)
#     for idx, row in terms.iterrows():
#         genes = row.used_genes
#         if len(genes) < 2:
#             continue
#         for i, g1 in enumerate(genes):
#             for g2 in genes[i + 1:]:
#                 pair = tuple(sorted((g1, g2)))
#                 gold_pair_to_complex[pair].add(idx)
    
#     # Precompute complex_ids as semicolon-separated strings in pairwise_df
#     pairs = [tuple(sorted((g1, g2))) for g1, g2 in zip(pairwise_df["gene1"], pairwise_df["gene2"])]
#     pairwise_df['complex_ids'] = [';'.join(map(str, sorted(gold_pair_to_complex.get(pair, set())))) for pair in pairs]
    
#     # Initialize AUC scores
#     auc_scores = {}
#     # Loop over each gene complex
#     for idx, row in tqdm(terms.iterrows()):
#         gene_set = set(row.used_genes)
#         if config["min_genes_per_complex_analysis"] > len(gene_set):  
#             continue
#         # Collect all row indices in the pairwise data where either gene belongs to the complex.
#         candidate_indices = bitarray(len(pairwise_df))
#         for gene in gene_set:
#             if gene in gene_to_pair_indices:
#                 candidate_indices[gene_to_pair_indices[gene]] = True
        
#         if not candidate_indices.any():
#             continue
        
#         # Select only the relevant pairwise comparisons.
#         selected_rows = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
#         sub_df = pairwise_df.iloc[selected_rows]
        
#         # Get current complex ID (assuming idx is the ID; adjust if row['ID'] is different)
#         complex_id = str(idx)  # Or str(row['ID']) if available
        
#         # Create true_label: 1 if complex_id in complex_ids (vectorized with str.contains)
#         #true_label = sub_df['complex_ids'].str.contains(complex_id, regex=False).astype(int)

#         # Inside the loop, for each complex:
#         # Inside the loop:
#         complex_id = str(idx)
#         # Use (?:^|;) and (?:;|$) to avoid capturing groups
#         pattern = r'(?:^|;)' + re.escape(complex_id) + r'(?:;|$)'
#         true_label = sub_df['complex_ids'].str.contains(pattern, regex=True).astype(int)
#         # Filter to keep verified negatives (complex_ids == "") or positives for this complex (true_label == 1)
#         complex_mask = (sub_df['complex_ids'] == "") | (true_label == 1)
        
#         # Use the masked true labels for AUPRC (avoids SettingWithCopyWarning)
#         predictions = true_label[complex_mask]
        
#         if predictions.sum() == 0:
#             continue
#         # Compute cumulative true positives and derive precision and recall.
#         true_positive_cumsum = predictions.cumsum()
#         precision = true_positive_cumsum / (np.arange(len(predictions)) + 1)
#         recall = true_positive_cumsum / true_positive_cumsum.iloc[-1]
        
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         auc_scores[idx] = metrics.auc(recall, precision)
    
#     # Add the computed AUC scores to the terms DataFrame.
#     terms["auc_score"] = pd.Series(auc_scores)
#     terms.drop(columns=["hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms



# it works quick but only maps 1 complex to each pair

# def pra_percomplex_old_type_filtering(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started - {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")
#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     pairwise_df = df.copy()
#     pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
#     pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")  
#     # Precompute a mapping from each gene to the row indices in the pairwise DataFrame where it appears.
#     gene_to_pair_indices = {}
#     for i, (gene_a, gene_b) in enumerate(zip(pairwise_df["gene1"], pairwise_df["gene2"])):
#         gene_to_pair_indices.setdefault(gene_a, []).append(i)
#         gene_to_pair_indices.setdefault(gene_b, []).append(i)  
#     # Initialize AUC scores (one for each complex) with NaNs.
#     #auc_scores = np.full(len(terms), np.nan)
#     auc_scores = {}
#     # Loop over each gene complex
#     for idx, row in tqdm(terms.iterrows()):
#         gene_set = set(row.used_genes)

#         if config["min_genes_per_complex_analysis"] > len(gene_set):  
#             continue
#         # Collect all row indices in the pairwise data where either gene belongs to the complex.
#         candidate_indices = bitarray(len(pairwise_df))
#         for gene in gene_set:
#             if gene in gene_to_pair_indices:
#                 candidate_indices[gene_to_pair_indices[gene]] = True      
#         if not candidate_indices.any():
#             continue     
#         # Select only the relevant pairwise comparisons.
#         selected_rows = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
#         sub_df = pairwise_df.iloc[selected_rows]
#         # A prediction is 1 if both genes in the pair are in the complex; otherwise 0.
#         predictions = (sub_df["gene1"].isin(gene_set) & sub_df["gene2"].isin(gene_set)).astype(int)
#         if predictions.sum() == 0:
#             continue
#         # Compute cumulative true positives and derive precision and recall.
#         true_positive_cumsum = predictions.cumsum()
#         precision = true_positive_cumsum / (np.arange(len(predictions)) + 1)
#         recall = true_positive_cumsum / true_positive_cumsum.iloc[-1]
        
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         auc_scores[idx] = metrics.auc(recall, precision)   
#     # Add the computed AUC scores to the terms DataFrame.
#     terms["auc_score"] = pd.Series(auc_scores)
#     terms.drop(columns=["hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms



# OLD
# def pra_percomplex(dataset_name, matrix, is_corr=False):
#     log.started(f"*** Per-complex PRA started for {dataset_name} ***")
#     config = dload("config")
#     terms = dload("tmp", "terms")
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")

#     if not is_corr:
#         matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present)
#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     # Precompute gene → row indices
#     gene_to_rows = {}
#     for i, (g1, g2) in enumerate(zip(df["gene1"], df["gene2"])):
#         gene_to_rows.setdefault(g1, []).append(i)
#         gene_to_rows.setdefault(g2, []).append(i)
#     aucs = np.full(len(terms), np.nan)
#     N = len(df)
#     for idx, row in tqdm(terms.iterrows()):
#         genes = set(row.used_genes)
#         if len(genes) < config["min_complex_size_for_percomplex"]:  # Skip small complexes
#             continue
#         # Get all row indices where either gene is in the complex
#         candidate_idxs = set()
#         for g in genes:
#             candidate_idxs.update(gene_to_rows.get(g, []))
#         candidate_idxs = sorted(candidate_idxs)
#         if not candidate_idxs:
#             continue
#         # Use only relevant rows for prediction
#         sub = df.loc[candidate_idxs]
#         preds = (sub["gene1"].isin(genes) & sub["gene2"].isin(genes)).astype(int)
#         if preds.sum() == 0:
#             continue
#         tp = preds.cumsum()
#         prec = tp / (np.arange(len(preds)) + 1)
#         recall = tp / tp.iloc[-1]
#         if len(recall) < 2 or recall.iloc[-1] == 0:
#             continue
#         aucs[idx] = metrics.auc(recall, prec)
#     terms["auc_score"] = aucs
#     terms.drop(columns=["list", "set", "hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.done(f"Per-complex PRA completed.")
#     return terms








# without greedy
# def complex_contributions(name):
#     log.info(f"Computing complex contributions for dataset: {name}")

#     pra = dload("pra", name)
#     terms = dload("tmp", "terms")
#     d = pra.query('prediction == 1').drop(columns=['gene1', 'gene2'])
#     results = {}
#     thresholds = [round(i, 2) for i in np.arange(1, 0.0001, -0.025)]
#     for cid in terms.ID.to_list():
#         arr = []
#         for threshold in thresholds:
#             r = d[d.complex_id == cid].query('precision >= @threshold')
#             arr.append(r.shape[0])
#         results[cid] = arr

#     r = pd.DataFrame(results, index=thresholds).T
#     t = terms[['ID', 'Name']].set_index('ID')
#     r['Name'] = r.index.map(t.Name)
#     r = r[list(reversed(list(r.columns)))]
#     r = r.reset_index(drop=True)
#     dsave(r, "complex_contributions", name)
#     log.info(f"Complex contributions computation completed for dataset: {name}")
#     return r






# # new
# def complex_contributions(name):
#     log.info(f"Computing complex contributions using R-style greedy logic for dataset: {name}")
#     pra = dload("pra", name)
#     terms = dload("common", "terms")
    
#     # Ensure pra is sorted by score descending
#     pra = pra.sort_values(by='score', ascending=False).reset_index(drop=True)
    
#     # Compute cumulative TP and precision if not present
#     pra['cumTP'] = pra['prediction'].cumsum()
#     pra['rank'] = pra.index + 1
#     pra['precision'] = pra['cumTP'] / pra['rank']
    
#     # R-style precision thresholds
#     prec_min = pra['precision'].min()
#     prec_max = pra['precision'].max()
#     precision_cutoffs = [round(prec_min, 3)]
#     cutoffs_range = np.arange(0.1, prec_max + 0.001, 0.025)
#     precision_cutoffs += [round(t, 3) for t in cutoffs_range if t > prec_min]
#     thresholds = sorted(set(precision_cutoffs))  # Ensure unique and sorted
    
#     results = {}
#     for t in thresholds:
#         if pra['precision'].max() < t:
#             continue
#         cand = pra[pra['precision'] >= t]
#         if cand.empty:
#             continue
#         k = cand.index.max()  # rightmost index where precision >= t
#         tp_target = pra.loc[k, 'cumTP']
#         # Find the smallest m where cumTP[m] >= tp_target
#         ind = pra[pra['cumTP'] >= tp_target].index.min()
#         if pd.isna(ind):
#             continue
#         # Select top (ind+1) rows
#         tmp = pra.iloc[0:ind + 1].copy()
#         # Filter for predicted positives (true == 1)
#         tmp = tmp[tmp['prediction'] == 1]
#         tmp = tmp[tmp["complex_id"].notnull()]
#         tmp["ID"] = tmp["complex_id"].apply(lambda ids: ";".join(str(int(i)) for i in ids if pd.notnull(i)))
#         # Now greedy logic
#         final_contrib = {}
#         while not tmp.empty:
#             all_ids = tmp["ID"].str.split(";").explode()
#             contrib = all_ids.value_counts()
#             if contrib.empty:
#                 break
#             top_id = contrib.idxmax()
#             final_contrib[top_id] = contrib[top_id]
#             tmp = tmp[~tmp["ID"].str.contains(rf"\b{top_id}\b", regex=True)]
#         for cid, count in final_contrib.items():
#             if cid not in results:
#                 results[cid] = [0] * len(thresholds)
#             results[cid][thresholds.index(t)] = count
    
#     # Add back gold standard complexes with 0 contribution
#     gold_ids = set(terms.index.astype(str))
#     all_ids = set(results.keys())
#     missing_ids = gold_ids - all_ids
#     for cid in missing_ids:
#         results[cid] = [0] * len(thresholds)
    
#     # Build result DataFrame
#     r = pd.DataFrame(results, index=thresholds).T
#     r['Name'] = r.index.astype(int).map(terms['Name'])
#     r = r[['Name'] + [c for c in r.columns if c != 'Name']]  # Name as first col
#     r = r[(r.drop(columns="Name").sum(axis=1) > 0)]
#     # Move ID to first column, keep Name second, then precision columns in order
#     dsave(r, "complex_contributions", name)
#     log.info(f"Greedy R-style complex contribution completed for dataset: {name}")
#     return r



# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"******************** {dataset_name} ********************")
#     log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")
#     config = dload("config")

#     terms_data = dload("tmp", "terms")
#     if terms_data is None or not isinstance(terms_data, pd.DataFrame):
#         raise ValueError("Expected 'terms' to be a DataFrame, but got None or invalid type.")
#     terms = terms_data
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")

#     if not is_corr:
#         matrix = perform_corr(matrix, config.get("corr_function"))
        
#     matrix = filter_matrix_by_genes(matrix, genes_present)

#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))

#     gold_pair_to_complex = defaultdict(list)
#     for idx, row in terms.iterrows():
#         genes = row.used_genes
#         if len(genes) < 2:
#             continue
#         for i, g1 in enumerate(genes):
#             for g2 in genes[i + 1:]:
#                 pair = tuple(sorted((g1, g2)))
#                 gold_pair_to_complex[pair].append(idx)


#     # Label predictions and complex IDs
#     complex_ids = []
#     predictions = []
#     for g1, g2 in zip(df["gene1"], df["gene2"]):
#         pair = tuple(sorted((g1, g2)))
#         ids = gold_pair_to_complex.get(pair, [])
#         if ids:
#             predictions.append(1)
#             complex_ids.append(ids)
#         else:
#             predictions.append(0)
#             complex_ids.append([])

#     df["prediction"] = predictions
#     df["complex_id"] = complex_ids

#     if df["prediction"].sum() == 0:
#         log.info("No true positives found in dataset.")
#         pr_auc = np.nan
#     else:
#         tp = df["prediction"].cumsum()
#         df["tp"] = tp
#         precision = tp / (np.arange(len(df)) + 1)
#         recall = tp / tp.iloc[-1]
#         pr_auc = metrics.auc(recall, precision)
#         df["precision"] = precision
#         df["recall"] = recall

#     log.info(f"PR-AUC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
#     dsave(df, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.done(f"Global PRA completed for {dataset_name}")
#     return df, pr_auc



# def compute_pra(df):
#     log.info("Calculating precision-recall and AUC score.")
#     if df.empty:
#         log.warning("Empty DataFrame encountered in compute_pra. Returning empty DataFrame.")
#         return df  
#     df["tp"] = df["prediction"].cumsum()
#     df.reset_index(drop=True, inplace=True)
#     df["precision"] = df["tp"] / (df.index + 1)
#     df["recall"] = df["tp"] / df["tp"].iloc[-1]
#     log.info("DONE: Calculating precision-recall AUC score.")
#     return df


# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"PRA computation started for {dataset_name}.")
#     genes_present_in_terms = dload("tmp", "genes_present_in_terms")
#     #terms_hash_table = dload("tmp", "terms_hash_table")
#     sorting_prefs = dload("input", "sorting")
#     sort_order = sorting_prefs.get(dataset_name, "high") 
#     if not is_corr: matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present_in_terms)
#     stack = binary(matrix)

#     log.info("Checking gene pairs against the gold standard.")
#     gene_pairs = list(zip(stack["gene1"], stack["gene2"]))
#     hashed_pairs = [hash(pair) for pair in gene_pairs]
#     stack["complex_id"] = [terms_hash_table.get(h, 0) for h in hashed_pairs]
#     stack["prediction"] = [1 if h in terms_hash_table else 0 for h in hashed_pairs]

#     annotated = stack.copy()
#     if sort_order == "low":
#         ann_sorted = quick_sort(annotated, ascending=True) 
#     else:
#         ann_sorted = quick_sort(annotated) 

#     pra = compute_pra(ann_sorted)
#     pr_auc = metrics.auc(pra.recall, pra.precision)
#     dsave(pra, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.info(f"PRA computation completed for {dataset_name} (Sorting: {sort_order}).")
#     return pra, pr_auc




# new but not seperated to functions (Build gold standard etc.)

# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"******************** {dataset_name} ********************")
#     log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")
#     terms_data = dload("tmp", "terms")
#     if terms_data is None or not isinstance(terms_data, pd.DataFrame):
#         raise ValueError("Expected 'terms' to be a DataFrame, but got None or invalid type.")
#     terms = terms_data.reset_index(drop=True)
#     genes_present = dload("tmp", "genes_present_in_terms")
#     sorting = dload("input", "sorting")
#     sort_order = sorting.get(dataset_name, "high")

#     if not is_corr:
#         matrix = perform_corr(matrix, "numpy")
        
#     matrix = filter_matrix_by_genes(matrix, genes_present)

#     log.info(f"Matrix shape: {matrix.shape}")
#     df = binary(matrix)
#     log.info(f"Pair-wise shape: {df.shape}")
#     df = quick_sort(df, ascending=(sort_order == "low"))
#     df = df.reset_index(drop=True)

#     # Build gold standard: map pair → complex ID
#     gold_pair_to_complex = {}
#     for idx, row in terms.iterrows():
#         genes = row.used_genes
#         if len(genes) < 2:
#             continue
#         for i, g1 in enumerate(genes):
#             for g2 in genes[i + 1:]:
#                 pair = tuple(sorted((g1, g2)))
#                 gold_pair_to_complex[pair] = idx

#     # Label predictions and complex IDs
#     complex_ids = []
#     predictions = []
#     for g1, g2 in zip(df["gene1"], df["gene2"]):
#         pair = tuple(sorted((g1, g2)))
#         if pair in gold_pair_to_complex:
#             predictions.append(1)
#             complex_ids.append(gold_pair_to_complex[pair])
#         else:
#             predictions.append(0)
#             complex_ids.append(0)
#     df["prediction"] = predictions
#     df["complex_id"] = complex_ids
#     if df["prediction"].sum() == 0:
#         log.info("No true positives found in dataset.")
#         pr_auc = np.nan
#     else:
#         tp = df["prediction"].cumsum()
#         df["tp"] = tp
#         precision = tp / (np.arange(len(df)) + 1)
#         recall = tp / tp.iloc[-1]
#         pr_auc = metrics.auc(recall, precision)
#         df["precision"] = precision
#         df["recall"] = recall
#     log.info(f"PR-AUC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
#     dsave(df, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.done(f"Global PRA completed for {dataset_name}")
#     return df, pr_auc

