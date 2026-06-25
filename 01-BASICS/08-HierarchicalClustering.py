import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
print(data)

linkage_data = linkage(data, method="ward", metric="euclidean")
dendrogram(linkage_data, get_leaves=True)
plt.show()

hierchial_cluster = AgglomerativeClustering(n_clusters=2, linkage="ward")
labels = hierchial_cluster.fit_predict(data)
plt.scatter(x, y, c=labels)
plt.show()
