"""
Basic usage example of the pythonFLEX package.
Demonstrates initialization, data loading, analysis, and plotting.
"""
#%%
import pythonflex as flex

inputs = {
    "Melanoma (63 Screens)": {
        "path": flex.get_example_data_path("melanoma_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
    "Liver (24 Screens)": {
        "path": flex.get_example_data_path("liver_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
    "Neuroblastoma (37 Screens)": {
        "path": flex.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
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
flex.plot_percomplex_scatter(n_top=20)
flex.plot_percomplex_scatter_bysize()
flex.plot_significant_complexes()
flex.plot_complex_contributions()


#%%
# Save results to CSV
flex.save_results_to_csv()













