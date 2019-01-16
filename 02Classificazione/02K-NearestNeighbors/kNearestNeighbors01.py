import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

'''
 Author: Cristiano Casadei
'''

# dati caricati da https://www.openml.org/d/28
X, Y = fetch_openml('optdigits', version=1, return_X_y=True)

# eseguo la codifica automatica delle label
labEnc = LabelEncoder()
Y_enc = labEnc.fit_transform(Y)

# divido in dataset train e test, usando un massimo di esempi per il train e di conseguenza il test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_enc, test_size=0.3, random_state=1234)

# eseguo la standardizzazione
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ciclo da 1 a 20 K
for K in range(1, 21):
    # eseguo la classificazione K-NN
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_std, Y_train)

    # eseguo predizione e calcolo confidenza su train
    Y_pred_train = knn.predict(X_train_std)
    Y_pred_train_proba = knn.predict_proba(X_train_std)

    # eseguo predizione e calcolo confidenza su test
    Y_pred = knn.predict(X_test_std)
    Y_pred_proba = knn.predict_proba(X_test_std)

    # valutiamo il modello con le metriche messe a disposizione da SciKitLearn
    print("K:", K)
    print("Accuracy: TRAIN =", "{0:.4f}".format(accuracy_score(Y_train, Y_pred_train)),
          "- TEST =", "{0:.4f}".format(accuracy_score(Y_test, Y_pred)))
    print("NLogLike: TRAIN =", "{0:.4f}".format(log_loss(Y_train, Y_pred_train_proba)),
          "- TEST =", "{0:.4f}".format(log_loss(Y_test, Y_pred_proba)))
    print()

# rieseguo la classificazione con K=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_std, Y_train)

# eseguo predizione e calcolo confidenza su test
Y_pred = knn.predict(X_test_std)

# visualizzo le cifre sbagliate
for i in range(0, len(X_test)):
    if Y_pred[i] != Y_test[i]:
        print("La cifra", Y_test[i], "è stata classificata come cifra", Y_pred[i])
        plt.imshow(X_test[i].reshape([8, 8]), cmap="gray")
        plt.show()
