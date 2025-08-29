


#%%

import os

# # Define specific cell line types you're interested in
DATA_DIR = "C:/Users/yd/Desktop/projects/_datasets/depmap/25Q2/subset/"

# Specific cell lines of interest with "_cell_lines" suffix removed
cell_line_files = [
    "soft_tissue_cell_lines.csv",
    "skin_cell_lines.csv", 
    "lung_cell_lines.csv",
    "head_and_neck_cell_lines.csv",
    "esophagus_stomach_cell_lines.csv",
    "pleura_cell_lines.csv"
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

inputs = {}
inputs['depmap'] = {
    "path": "C:/Users/yd/Desktop/projects/_datasets/depmap/25Q2/gene_effect.csv",
    "sort": "high"
}

# Print the resulting inputs dictionary
print("Configured inputs:")
for key, value in inputs.items():
    print(f"  {key}: {value['path']}")

