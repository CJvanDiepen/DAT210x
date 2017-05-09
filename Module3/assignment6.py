import pandas as pd
import matplotlib.pyplot as plt


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv(r"C:\Users\Sjaak\Documents\DAT210x\Module3\Datasets\wheat.data", index_col=0)

#
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# 
# .. your code here ..


#
# TODO: Compute the correlation matrix of your dataframe
# 
# .. your code here ..
corr_matr = df.corr()

#
# TODO: Graph the correlation matrix using imshow or matshow
# 
# .. your code here ..
plt.figure()
plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()


plt.show()


