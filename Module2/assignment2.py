import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#
# .. your code here ..
tutorial_df = pd.read_csv(r'C:\Users\diepencjv\repos\DAT210x\Module2\Datasets\tutorial.csv')


# TODO: Print the results of the .describe() method
#
# .. your code here ..
print(tutorial_df)
print(tutorial_df.describe())


# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..
tutorial_df.loc[2:4, 'col3']
tutorial_df.iloc[2:3, 3]
tutorial_df.ix[0:10, 'col3']
tutorial_df[2:4,3]
