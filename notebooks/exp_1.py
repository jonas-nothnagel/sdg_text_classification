# %%
'''basics'''
import pandas as pd
import transformers
print('transformers: {}'.format(transformers.__version__))
# %%
df= pd.read_csv('../data/raw/osdg-community-dataset-v21-09-30.csv',sep='\t')
df.head(5)
# %%
print('average text length: ', df.text.str.split().str.len().mean().round(2))
print('stdev text length: ', df.text.str.split().str.len().std().round(2))
print('max text length: ', df.text.str.split().str.len().max().round(2))

if df.text.str.split().str.len().mean() < 256:
    print("suitable for standard transformer models!")
# %%
df
# %%
