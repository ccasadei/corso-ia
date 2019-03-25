import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd

# preparo la visualizzazione
plt.rcParams["figure.figsize"] = (12, 8)
sns.set()

# creo un dataset random, ignoro il target
X, _ = make_blobs(n_samples=50, centers=5, cluster_std=0.5, random_state=1234)

# visualizzo il dataset così creato
plt.scatter(X[:, 0], X[:, 1], s=70)
plt.show()

# costruisco una matrice di associazioni (linkage), usando il Ward's linkage
link_matrix = linkage(X, method="ward")

# visualizzo la matrice di linkage
pd_lnk_mtx = pd.DataFrame(link_matrix, columns=["cluster ID_A", "cluaster ID_B", "distanza", "numero dati"])
print(pd_lnk_mtx)

# visualizzo il dendrogramma
dendrogram(link_matrix)
plt.show()

# ora che ho idea della soglia da utilizzare, posso creare il modello di clustering agglomerativo
# NOTA: in SciKitLearn è necessario indicare subito il numero di cluster, quindi si va a perdere
# l'utilità del dendrogramma...
agglom_clustering = AgglomerativeClustering(n_clusters=5)
# in un unico passaggio addestro il modello ed elaboto la predizione
y = agglom_clustering.fit_predict(X)

# visualizzo la clusterizzazione ottenuta
plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap="viridis", edgecolors="black")
plt.show()
