
# %%
import pandas as pd

df = pd.read_csv("../../../../datasets/depmap/24Q4/CRISPRGeneEffect.csv",index_col=0)
model = pd.read_csv("../../../../datasets/depmap/24Q4/Model.csv",index_col=0)

df.columns = df.columns.str.split(" \\(").str[0]
df = df.T

#%%

# %%
# get ModelID of selected disease for example OncotreePrimaryDisease==Melanoma
melanoma = model[model.OncotreePrimaryDisease=="Melanoma"].index.unique().values
liver = model[model.OncotreeLineage=="Liver"].index.unique().values
neuroblastoma = model[model.OncotreePrimaryDisease=="Neuroblastoma"].index.unique().values

# %%
# mel.index is model ids, filter that ids in the columns of df
mel_df = df.loc[:,df.columns.isin(melanoma)]
liver_df = df.loc[:,df.columns.isin(liver)]
neuro_df = df.loc[:,df.columns.isin(neuroblastoma)]


# %%
mel_df.to_csv("melanoma.csv")
liver_df.to_csv("liver.csv")
neuro_df.to_csv("neuroblastoma.csv")
df.to_csv("depmap_geneeffect_all_cellines.csv")
# %%
