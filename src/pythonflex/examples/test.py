#%%
import pythonflex as flex
import os

# # Define specific cell line types you're interested in
DATA_DIR = "C:/Users/yd/Desktop/projects/_datasets/depmap/25Q2/subset/"

# Specific cell lines of interest with "_cell_lines" suffix removed
cell_line_files = [
    "soft_tissue_cell_lines.csv",
    "skin_cell_lines.csv", 
    # "lung_cell_lines.csv",
    # "head_and_neck_cell_lines.csv",
    # "esophagus_stomach_cell_lines.csv",
]

inputs = {}

# Create inputs dict with shortened names (removing "_cell_lines" suffix)
for filename in cell_line_files:
    # Remove .csv extension and _cell_lines suffix
    key = filename.replace("_cell_lines.csv", "")
    full_path = os.path.join(DATA_DIR, filename)
    
    inputs[key] = {
        "path": full_path,
        "sort": "high"
    }

inputs['depmap'] = {
    "path": "C:/Users/yd/Desktop/projects/_datasets/depmap/25Q2/gene_effect.csv",
    "sort": "high"
}

# Print the resulting inputs dictionary
print("Configured inputs:")
for key, value in inputs.items():
    print(f"  {key}: {value['path']}")



default_config = {
    "min_genes_in_complex": 2,
    "min_genes_per_complex_analysis": 2,
    "output_folder": "25q2_min_genes_2",
    "gold_standard": "CORUM",
    "color_map": "RdYlBu",
    "jaccard": True,
    "plotting": {
        "save_plot": True,
        "output_type": "pdf",
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
    pra = flex.pra(name, dataset, is_corr=False)
    fpc = flex.pra_percomplex(name, dataset, is_corr=False) 
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









#%%


