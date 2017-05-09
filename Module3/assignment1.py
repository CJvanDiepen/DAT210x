import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv(r"C:\Users\Sjaak\Documents\DAT210x\Module3\Datasets\wheat.data")

#
# TODO: Create a slice of your dataframe (call it s1)
# that only includes the 'area' and 'perimeter' features
# 
# .. your code here ..
s1 = df.ix[:,['area','perimeter']]

#
# TODO: Create another slice of your dataframe (call it s2)
# that only includes the 'groove' and 'asymmetry' features
# 
# .. your code here ..
s2 = df.ix[:, ['groove', 'asymmetry']]

#
# TODO: Create a histogram plot using the first slice,
# and another histogram plot using the second slice.
# Be sure to set alpha=0.75
# 
# .. your code here ..
s1.plot.hist(alpha=.75)
s2.plot.hist(alpha=.75)


# Display the graphs:
plt.show()

