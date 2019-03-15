import sklearn.datasets as ds
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scikitplot as skplt

z_scaler = StandardScaler()
data = ds.load_breast_cancer()['data'];

z_data = z_scaler.fit_transform(data)
pca_trafo = PCA().fit(z_data);


plt.semilogy(pca_trafo.explained_variance_ratio_, '--o');
plt.semilogy(pca_trafo.explained_variance_ratio_.cumsum(), '--o');
plt.show()
