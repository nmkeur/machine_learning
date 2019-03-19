import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

np.logspace(2,9, num=4)
x = np.arange(-1, 1, 0.01)
x[-1]

np.logspace(-17,1,num=20,base=10)


scaler = StandardScaler()

datafile_path = 'norm_reads.csv'
norm_data = pd.read_csv(datafile_path, index_col=0)
y_value = norm_data.patientgroup

norm_datat = norm_data.transpose()

# Select all column from the dataframe except
# the first (sampleID) and last (patientgroup)
x_value = norm_data[norm_data.columns[0:12490]]

x_train, x_test, y_train, y_test = train_test_split(x_value, y_value,
                                        test_size=0.2, shuffle=True, stratify=y_value)
# Fit on training set only.

from sklearn.datasets import load_iris


x_value.head()
x_valuet = x_value.transpose()
x_valuet.head()

df_scaled = pd.DataFrame(scaler.fit_transform(norm_data),index=norm_data.index, columns = norm_data.columns)
df_scaled

# Make an instance of the Model
pca = PCA(0.99)
pca.fit(x_valuet)

pca.n_components_
pca.components_
