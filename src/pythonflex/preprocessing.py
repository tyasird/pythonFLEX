import os
import pandas as pd
import numpy as np
from .utils import dsave, dload
from tqdm import tqdm
from .logging_config import log
from importlib import resources
from pathlib import Path
tqdm.pandas()


def get_example_data_path(filename: str):
    return resources.files("pythonflex.data").joinpath("dataset").joinpath(filename)


def _load_file(filepath, ext):
    loaders = {
        ".csv": lambda f: pd.read_csv(f, index_col=0),
        ".xlsx": lambda f: pd.read_excel(f, index_col=0),
        ".parquet": pd.read_parquet,
        ".p": pd.read_parquet
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loaders[ext](filepath)


def load_datasets(files, continue_with_common_genes=False):
    preprocessing = dload("config")["preprocessing"]
    data_dict= {}     

    for filename, meta in files.items():
        if isinstance(meta, pd.DataFrame):
            df = meta
        elif isinstance(meta, dict):
            filepath = meta["path"]
            if isinstance(filepath, pd.DataFrame):
                df = filepath
            else:
                ext = os.path.splitext(filepath)[1]
                df = _load_file(filepath, ext)
        else:
            raise ValueError(f"Unsupported data structure for '{filename}': {type(meta)}")

        df.index = df.index.str.split().str[0]
        if preprocessing.get('normalize'):
            log.info(f"{filename}: Normalization.")
            df = (df - df.mean()) / df.std(ddof=0)

        if preprocessing.get('drop_na'):
            log.info(f"{filename}: Dropping missing values.")
            df = df.dropna(how="any")

        if preprocessing.get('fill_na'):
            log.info(f"{filename}: Filling missing values with column mean.")
            #df = df.T.fillna(df.mean(axis=1)).T
            df = data_imputation(df)

            
        data_dict[filename] = df

    common_genes = get_common_genes(data_dict)
    if continue_with_common_genes:
        log.info(f"Continuing with common genes: {len(common_genes)}")
        for filename, df in data_dict.items():
            if df.index.isin(common_genes).any():
                data_dict[filename] = df.loc[common_genes]
    
    dsave({
        "datasets": data_dict,
        "sorting": {
            k: v.get("sort", "high") if isinstance(v, dict) else "high"
            for k, v in files.items()
        }
    }, "input")
    log.done(f"Datasets loaded.")
    return data_dict, common_genes  




def drop_bad_samples(df, max_na=0.1):
    total_elements = df.shape[0] * df.shape[1]
    percent_nan = np.isnan(df.values).sum() / total_elements if total_elements > 0 else 0
    has_nan_per_sample = np.isnan(df.values).any(axis=0) # how many samples has NA.

    log.info(f"Total: {total_elements}, Percent NaN: {percent_nan:.2%}, Samples with NaN: {np.sum(has_nan_per_sample)} / {df.shape[1]}")

    num_genes = df.shape[0]  # E.g., 1178 (total rows/genes)
    na_per_sample = np.isnan(df.values).sum(axis=0) / num_genes  # Fraction NA per sample (column)
    good_samples = na_per_sample <= max_na  # Keep if <=10% NA
    data_filtered = df.loc[:, good_samples]  # Drop bad samples (those >10% NA)

    log.info(f"Filtered samples: {data_filtered.shape[1]} (removed {df.shape[1] - data_filtered.shape[1]} samples with >{max_na*100:.0f}% NAs)")
    return data_filtered



def data_imputation(df):
    log.info("Imputing missing values with gene means ...")
    gene_means = np.nanmean(df.values, axis=1)  # 1D array: means per gene
    data_values = df.values.copy()
    rows, cols = np.where(np.isnan(data_values))
    if len(rows) > 0:
        data_values[rows, cols] = np.take(gene_means, rows)

    df = pd.DataFrame(data_values, index=df.index, columns=df.columns)
    log.info(f"Data after imputation: {df.shape[0]} genes, {df.shape[1]} samples")
    return df




def get_common_genes(datasets):
    log.started("Finding common genes across datasets.")
    gene_sets = [set(df.index) for df in datasets.values()]
    common_genes = set.intersection(*gene_sets)
    log.done(f"Common genes found: {len(common_genes)}")
    dsave(common_genes, "common", "common_genes")
    return list(common_genes)


def filter_matrix_by_genes(matrix, genes_present_in_terms):
    log.started("Filtering matrix using genes present in terms.")
    genes = matrix.index.intersection(genes_present_in_terms)
    matrix = matrix.loc[genes, genes]
    log.done(f"Filtering matrix: {matrix.shape}")
    return matrix.loc[genes, genes]
    



def load_gold_standard():
    
    config = dload("config") 
    common_genes = dload("common", "common_genes")

    gold_standard_source = config['gold_standard']
    log.started(f"Loading gold standard: {gold_standard_source}, Min complex size: {config['min_genes_in_complex']}, Jaccard filtering: {config['jaccard']}")
    if not common_genes:
        raise ValueError("Common genes not found.")

    # Define gold standard file paths for predefined sources
    gold_standard_files = {
        "CORUM": "gold_standard/CORUM.parquet",
        "GOBP": "gold_standard/GOBP.parquet",
        "PATHWAY": "gold_standard/PATHWAY.parquet"
    }

    if gold_standard_source in gold_standard_files:
        # Load predefined gold standard from package resources
        filename = gold_standard_files[gold_standard_source]
        filename_path = resources.files("pyflex.data").joinpath(filename)
        if not filename_path.exists():  # Check if the file exists
            raise ValueError(f"Invalid Gold Standard type: {gold_standard_source}. File not found.")
        terms = pd.read_parquet(filename_path)  # type: ignore
    elif Path(gold_standard_source).suffix.lower() == '.csv':
        # Load user-provided custom gold standard from CSV file
        filename_path = Path(gold_standard_source)
        if not filename_path.exists():
            raise ValueError(f"Custom gold standard CSV file not found: {gold_standard_source}")
        log.done(f"Loading custom gold standard from CSV: {gold_standard_source}")
        terms = pd.read_csv(filename_path)  
    else:
        raise ValueError(f"Invalid gold standard source: {gold_standard_source}. Must be one of {list(gold_standard_files.keys())} or a path to a .csv file.")

    common_genes_set = set(common_genes)
    terms["used_genes"] = terms["Genes"].apply(lambda x: list(set(x.split(";")) & common_genes_set))
    terms["n_used_genes"] = terms["used_genes"].apply(len)
    log.info(f"Applying min_genes_in_complex filtering: {config['min_genes_in_complex']}")
    terms = terms[terms["n_used_genes"] >= config['min_genes_in_complex']]
    terms["hash"] = terms["used_genes"].apply(lambda x: [hash(i) for i in x])

    if config['jaccard']:
        log.info("Applying Jaccard filtering. Remove terms with identical gene sets.")
        terms = filter_duplicate_terms(terms)


    genes_present_in_terms = list(set(terms["used_genes"].explode().unique()) & common_genes_set)
    # if there is column called "ID", set it as index
    if "ID" in terms.columns:
        terms = terms.set_index("ID")

    dsave(terms, "common", "terms")
    dsave(genes_present_in_terms, "common", "genes_present_in_terms")
    log.done("Gold standard loading completed.")
    return terms, genes_present_in_terms





def filter_duplicate_terms(terms):
    log.started("Filtering duplicate terms using optimized method.")
    
    # Precompute frozen gene sets and hash them
    terms = terms.copy()
    terms["gene_set"] = terms["used_genes"].map(lambda x: frozenset(x))
    
    # Group by identical gene sets
    grouped = terms.groupby("gene_set", sort=False)
    
    # Identify duplicate clusters (groups with >1 term)
    duplicate_clusters = []
    for _, group in grouped:
        if len(group) > 1:
            duplicate_clusters.append(group["ID"].values)
    
    # Determine which IDs to keep (smallest ID in each duplicate cluster)
    keep_ids = set(terms["ID"])
    for cluster in duplicate_clusters:
        sorted_ids = sorted(cluster)
        keep_ids.difference_update(sorted_ids[1:])  # Remove all but smallest ID
    
    # Filter and clean up
    filtered = terms[terms["ID"].isin(keep_ids)].copy()
    filtered.drop(columns=["gene_set"], inplace=True)
    
    log.done(f"{len(terms) - len(filtered)} terms removed due to identical gene sets.")
    return filtered
