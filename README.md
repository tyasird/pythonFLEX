# pythonFLEX

🧬 **pythonFLEX** is a benchmarking toolkit for evaluating CRISPR screen results against biological gold standards. It provides precision-recall analysis using reference gene sets from CORUM protein complexes, Gene Ontology Biological Processes (GO-BP), KEGG pathways, and other curated resources. The toolkit computes gene-level and complex-level performance metrics, helping researchers systematically assess the biological relevance and resolution of their CRISPR screening data.


---

## 🔧 Features

- Precision-recall curve generation for ranked gene lists

- Evaluation using CORUM complexes, GO terms, pathways

- Complex-level resolution analysis and visualization

- Easy integration into CRISPR screen workflows

---

## 📦 Installation

Suggested to use Python version `3.10` with `virtual env`.

Create `venv`

```bash
conda create -n p310 python=3.10
conda activate p310
pip install uv
```

Install pythonFLEX via pip

``` bash
uv pip install pythonflex
```

or 

```bash
pip install pythonflex
```

or Install pythonFLEX via git (to develop package in local)

```bash
git clone https://github.com/tyasird/pythonFLEX.git
cd pythonFLEX
uv pip install -e .
```



---

## 🚀 Quickstart

```python

import pythonflex as flex

inputs = {
    "Melanoma (63 Screens)": {
        "path": flex.get_example_data_path("melanoma_cell_lines_500_genes.csv"), 
        "sort": "high",
        "color": "#FF0000"
    },
    "Liver (24 Screens)": {
        "path": flex.get_example_data_path("liver_cell_lines_500_genes.csv"), 
        "sort": "high",
        "color": "#FFDD00"
    },
    "Neuroblastoma (37 Screens)": {
        "path": flex.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"), 
        "sort": "high",
        "color": "#FFDDDD"
    },
}



default_config = {
    "min_genes_in_complex": 0,
    "min_genes_per_complex_analysis": 3,
    "output_folder": "CORUM",
    "gold_standard": "CORUM",
    "color_map": "BuGn",
    "jaccard": False,
    "use_common_genes": False,  # Set to False for individual dataset-gold standard intersections
    "plotting": {
        "save_plot": True,
        "output_type": "png",
    },
    "preprocessing": {
        "fill_na": True,
        "normalize": False,
    },
    "corr_function": "numpy",
    "logging": {  
        "visible_levels": ["DONE"]  
        # "PROGRESS", "STARTED", ,"INFO","WARNING"
    }
}

# Initialize logger, config, and output folder
flex.initialize(default_config)

# Load datasets and gold standard terms
data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_gold_standard()

# Run analysis
for name, dataset in data.items():
    pra = flex.pra(name, dataset, is_corr=False)
    fpc = flex.pra_percomplex(name, dataset, is_corr=False) 
    cc = flex.complex_contributions(name)
    flex.mpr_prepare(name)  


#%%
# Generate plots
flex.plot_precision_recall_curve()
flex.plot_auc_scores()
flex.plot_significant_complexes()
flex.plot_percomplex_scatter(n_top=20)
flex.plot_percomplex_scatter_bysize()
flex.plot_complex_contributions()
##
flex.plot_mpr_tp_multi()
flex.plot_mpr_complexes_multi()

# Save results to CSV
flex.save_results_to_csv()


```

---

## 📂 Examples

- [src/pythonflex/examples/basic_usage.py](src/pythonflex/examples/basic_usage.py)

---

## 📃 License

MIT
