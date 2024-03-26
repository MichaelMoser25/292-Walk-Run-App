import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## Extract data from csv file
dataset = pd.read_csv('test.csv')

## Seperate labels from data
data = dataset.iloc[:, :-1]
labels = dataset.iloc[:, -1]

## Plot a 4 by 4 grid
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))

data.hist(ax=ax.flatten()[0:13]) ## Data has 13 columns

## Render plot
fig.tight_layout()
plt.show()



## Seperate labels from data
dataset = pd.read_csv("unclean-wine-quality.csv")
dataset = dataset.iloc[:, 1:-1]

## Clean the data
nanIndices = np.where(pd.isnull(dataset))
print("The indices of NaNs are:\n ", nanIndices)
numNans = dataset.isna().sum().sum()
print("The total number of NaNs is:", numNans)

print("\n")

dashIndices = np.where(dataset == "-")
print("The indices of \"-\" are:\n", dashIndices)
numDashes = np.sum(dataset.values == "-").sum()
print("The total number of \"-\" is: ", numDashes)

dataset.mask(dataset == "-", other=np.nan, inplace=True)
dataset = dataset.astype('float64')


## Replace NaN values with constants
fill_values = {
    'fixed acidity': 0,
    'volatile acidity': 0,
    'citric acid': 0,
    'residual sugar': 0,
    'chlorides': 1,
    'free sulfur dioxide': 0,
    'total sulfur dioxide': 0,
    'density': 0,
    'pH': 1,
    'sulphates': 1,
    'alcohol': 0
}

dataset.fillna(fill_values, inplace=True)
numNans = dataset.isna().sum().sum()
print("The number of NaNs is: ", numNans)


## For example, if a NaN exists under the column ‘pH’, it should be replaced with 1. After replacing
## all NaNs with the specified values, print the total number of NaNs in the dataset again.


## Sample and hold filling
dataset.fillna(method='ffill', inplace=True)
print("[16, 0]: ", dataset.iloc[16, 0])
print("[17, 0]: ", dataset.iloc[17, 0])

## Linear interpolation