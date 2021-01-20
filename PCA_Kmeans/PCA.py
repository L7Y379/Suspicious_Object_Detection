from sklearn.decomposition import PCA
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=1)
reduced_x=pca.fit_transform(X)

#PCA(copy=True, n_components=2, whiten=False)
print(reduced_x)
print(pca.explained_variance_ratio_)
