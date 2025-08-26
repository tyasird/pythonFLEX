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



default_config = {
    "min_genes_in_complex": 2,
    "min_genes_per_complex_analysis": 2,
    "output_folder": "output",
    "gold_standard": "GOBP",
    "color_map": "RdYlBu",
    "jaccard": True,
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
        "visible_levels": ["DONE","STARTED"]  # "PROGRESS", "STARTED", ,"INFO","WARNING"
    }
}


# Initialize logger, config, and output folder
flex.initialize(default_config)

# Load datasets and gold standard terms
data, _ = flex.load_datasets(inputs)
terms, genes_in_terms = flex.load_gold_standard()

# Run analysis
for name, dataset in data.items():
    df, pr_auc = flex.pra(name, dataset)
    fpc = flex.pra_percomplex(name, dataset, is_corr=False) 
    cc = flex.complex_contributions(name)

# Generate plots
flex.plot_auc_scores()
flex.plot_precision_recall_curve()
flex.plot_percomplex_scatter()
flex.plot_percomplex_scatter_bysize()
flex.plot_significant_complexes()
flex.plot_complex_contributions()

# Save Result CSVspyflex.save_results_to_csv()
flex.save_results_to_csv()


```

---

## 📂 Examples

- [src/pythonflex/examples/basic_usage.py](src/pythonflex/examples/basic_usage.py)

---

## 📃 License

MIT
