#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append(r"C:\Users\diepencjv\repos\DAT210x\Module4")
import assignment2_helper as helper

#%% only use numeric features
#features = 'numeric'
features = 'all'

df = pd.read_csv(r"C:\Users\diepencjv\repos\DAT210x\Module4\Datasets\kidney_disease.csv", index_col='id')
df = df.dropna()
labels = ['red' if i=='ckd' else 'blue' for i in df.classification]
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

if features == 'numeric':
    df = df.drop(['classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis=1)

if features == 'all':
    df = df.drop(['classification'], axis=1)
    df = pd.get_dummies(df,columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

#%% scale features
#print(df.var())
df = helper.scaleFeatures(df)

#%% use PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(df)
T = pca.transform(df)

#%% visualize processed data
scaleFeatures = True
plt.figure()
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()

#%%
