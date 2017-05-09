import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv(r"C:\Users\Sjaak\Documents\DAT210x\Module3\Datasets\wheat.data", index_col=0)
df = pd.read_csv(r"C:\Users\Sjaak\Documents\DAT210x\Module3\Datasets\wheat.data")


#
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# Also get rid of the 'area' and 'perimeter' features
# 
# .. your code here ..
df = df.drop('area', axis=1)
df = df.drop('perimeter', axis=1)

#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
parallel_coordinates(df, 'wheat_type', alpha=.4)


plt.show()


#%%
from pandas.tools.plotting import andrews_curves

plt.figure()
andrews_curves(df, 'wheat_type', alpha=0.4)
