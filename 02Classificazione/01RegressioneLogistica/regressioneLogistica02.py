import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

'''
 Author: Cristiano Casadei
'''

# dati caricati da https://www.openml.org/d/554
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)

# limitiamo il dataset
dataset_limit = 5000
X = X[:dataset_limit]
Y = Y[:dataset_limit]

# diamo uno sguardo ai dati di input e di output
print("Dimensioni dei dati di input:", X.shape)
print("Classi di output:", np.unique(Y))

# visualizziamo le 10 cifre decimali per vedere come sono fatte
for i in range(0, 10):
    digit = X[Y == str(i)][0].reshape([28, 28])
    ll_plot = plt.subplot(2, 5, i + 1)
    ll_plot.imshow(digit, cmap="gray")
plt.show()

# eseguo la codifica automatica delle label
labEnc = LabelEncoder()
Y_enc = labEnc.fit_transform(Y)

# visualizzo le label codificate
print("Label codificate:", np.unique(Y_enc))

# confronto label originali con label codificate
for i in range(10):
    # prendo il primo indice di Y_enc che contiene il valore corrente di i
    indice_label = np.where(Y_enc == i)[0][0]
    # visualizzo la label originale in stessa posizione di quella codificata per un confronto
    print("Label originale:", Y[indice_label], type(Y[indice_label]),
          "- Label codificata:", Y_enc[indice_label], type(Y_enc[indice_label]))

# divido in dataset train e test, usando un massimo di esempi per il train e di conseguenza il test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_enc, test_size=0.3, random_state=1234)

# eseguo la standardizzazione
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# eseguo una regressione logistica
logReg = LogisticRegression()
logReg.fit(X_train_std, Y_train)

# eseguo predizione e calcolo confidenza
Y_pred = logReg.predict(X_test_std)
Y_pred_proba = logReg.predict_proba(X_test_std)

cnf_matrix = confusion_matrix(Y_test, Y_pred)

# visualizzo la matrice di confusione in modo grafico
class_names = np.unique(Y_enc)
_, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matrice di confusione regressione logistica default', y=1.1)
plt.ylabel('Classi reali')
plt.xlabel('Classi predette')
plt.show()

# valutiamo il modello con le metriche messe a disposizione da SciKitLearn
print("Risultati regressione logistica default")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
# uso il parametro average='weighted' perchè si tratta di una classificazione multiclasse
print("Precision:", precision_score(Y_test, Y_pred, average='weighted'))
print("Recall:", recall_score(Y_test, Y_pred, average='weighted'))
print("Neg. Log-Likelihood:", log_loss(Y_test, Y_pred_proba))
print()

# ripeto modificando i parametri della regressione logistica
logReg = LogisticRegression(multi_class='multinomial',
                            penalty='l2', solver='sag')
logReg.fit(X_train_std, Y_train)

# eseguo predizione e calcolo confidenza
Y_pred = logReg.predict(X_test_std)
Y_pred_proba = logReg.predict_proba(X_test_std)

cnf_matrix = confusion_matrix(Y_test, Y_pred)

# visualizzo la matrice di confusione in modo grafico
class_names = np.unique(Y_enc)
_, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matrice di confusione regressione logistica modificata', y=1.1)
plt.ylabel('Classi reali')
plt.xlabel('Classi predette')
plt.show()

# valutiamo il modello con le metriche messe a disposizione da SciKitLearn
print("Risultati regressione logistica modificata")
print("Accuracy:", accuracy_score(Y_test, Y_pred))
# uso il parametro average='weighted' perchè si tratta di una classificazione multiclasse
print("Precision:", precision_score(Y_test, Y_pred, average='weighted'))
print("Recall:", recall_score(Y_test, Y_pred, average='weighted'))
print("Neg. Log-Likelihood:", log_loss(Y_test, Y_pred_proba))
