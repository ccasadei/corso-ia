import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.datasets import make_moons

plt.rcParams["figure.figsize"] = (14, 10)

# creo un dataset decisamente non "sferico", usando una apposita funzione di SciKitLearn
X, _ = make_moons(n_samples=400, noise=0.1, random_state=1234)
plt.scatter(X[:, 0], X[:, 1], s=300)
plt.show()

# istanzio la classe di clustering DBSCAN
dbscan = DBSCAN(eps=0.15, min_samples=3)
# eseguo fitting e predizione in una volta sola
y_dbscan = dbscan.fit_predict(X)

# visualizzo il risultato
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap="viridis", s=300, edgecolors="black")
plt.show()

# verifico il risultato del k-means
km = KMeans(init="k-means++", n_clusters=2)
y_kmeans = km.fit_predict(X)

# visualizzo il risultato
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=300, edgecolors="black")
plt.show()

# verifico con cluster gerarchico
agglom_clustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
# in un unico passaggio addestro il modello ed elaboto la predizione
y_agglomerative = agglom_clustering.fit_predict(X)

# visualizzo il risultato
plt.scatter(X[:, 0], X[:, 1], c=y_agglomerative, cmap="viridis", s=300, edgecolors="black")
plt.show()

