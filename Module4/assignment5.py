import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#%%
samples = []

import os

basedir = r'C:\Users\Sjaak\Documents\DAT210x\Module4\Datasets\ALOI\32'
files = os.listdir(basedir)

for fname in files: 
    img = misc.imread(os.path.join(basedir,fname))
    samples.append(img.reshape(-1))

df = pd.DataFrame(samples)
    
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 
#%%
from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=6, n_components=3)
iso.fit(samples)
T = iso.transform(samples)

def Plot2D(T, title, x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)

def Plot3D(T, title, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Component: {0}'.format(x))
    ax.set_ylabel('Component: {0}'.format(y))
    ax.set_zlabel('Component: {0}'.format(z))
    ax.scatter(T[:,x],T[:,y],T[:,z], marker='.',alpha=0.7)
    
Plot2D(T, title='ISO2D', x=0, y=1)
Plot2D(T, title='ISO2D', x=1, y=2)
Plot2D(T, title='ISO2D', x=0, y=2)

Plot3D(T, title='ISO3D', x=0, y=1, z=2)

for n in [5, 4, 3, 2, 1]:
    iso = Isomap(n_neighbors=n, n_components=3)
    iso.fit(samples)
    T = iso.transform(samples)
    
    Plot3D(title='ISO3D, neighbors: %d' % n, x=0, y=1, z=2)


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 



#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

