"""
Basic usage example of the pythonFLEX package.
Demonstrates initialization, data loading, analysis, and plotting.
"""
#%%
import pythonflex as flex

inputs = {
    "SNF": {
        "path":  "C:/Users/yd/Desktop/projects/datasets/fused_similarity_network.csv",
        "sort": "high"
    },
    "miss_SNF": {
        "path":  "C:/Users/yd/Desktop/projects/datasets/miss_snf_fused_similarity_network.csv",
        "sort": "high"
    }
}

#%%

default_config = {
    "min_genes_in_complex": 0,
    "min_genes_per_complex_analysis": 3,
    "output_folder": "output",
    "gold_standard": "CORUM",
    "color_map": "RdYlBu",
    "jaccard": True,
    "plotting": {
        "save_plot": True,
        "output_type": "PNG",
    },
    "preprocessing": {
        "fill_na": True,
        "normalize": False,
    },
    "corr_function": "numpy",
    "logging": {  
        "visible_levels": ["DONE","STARTED"]  # "PROGRESS", "STARTED", ,"INFO","WARNING"
    }
}

# Initialize logger, config, and output folder
flex.initialize(default_config)

# Load datasets and gold standard terms

data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_gold_standard()


#%%
# Run analysis
for name, dataset in data.items():
    df, pr_auc = flex.pra(name, dataset, is_corr=True)
    fpc = flex.pra_percomplex(name, dataset, is_corr=True) 
    cc = flex.complex_contributions(name)


#%%
# Generate plots
flex.plot_auc_scores()
flex.plot_precision_recall_curve()
flex.plot_percomplex_scatter()
flex.plot_percomplex_scatter_bysize()
flex.plot_significant_complexes()
flex.plot_complex_contributions()


#%%
# Save results to CSV
flex.save_results_to_csv()













# %%
import os
import glob

inputs = {
    "depmap all": {
        "path":  "../../../../datasets/depmap/24Q4/depmap_geneeffect_all_cellines.csv",
        "sort": "high"
    }
}

# Now auto-discover the rest of the CSVs in the folder
DATA_DIR = "../../../../datasets/depmap/24Q4/subset/"
for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):

    # Derive the key name from filename (without extension)
    key = os.path.splitext(os.path.basename(path))[0]
    inputs[key] = {
        "path": path,
        "sort": "high"
    }

# inputs now has "depmap all" first, then one entry per CSV in DATA_DIR
print(inputs)