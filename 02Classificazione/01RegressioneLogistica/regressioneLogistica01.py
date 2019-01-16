import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
 Author: Cristiano Casadei
'''

# carico il dataset (contiene già gli header, quindi non li devo specificare)
pima = pd.read_csv("./diabetes.csv")

# diamo un'occhiata al contenuto
print(pima.head())

# diamo un'occhiata alle classi di Outcome (la nostra proprietà di uscita)
print(pima["Outcome"].unique())

# preparo la matrice di input ed il vettore di output
X = pima.drop("Outcome", axis=1).values
Y = pima["Outcome"].values

# suddivido il dataset in dataset di train e di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# istanzio la classe di standardizzazione e standardizzo i dataset
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# istanzio la classe di regressione logistica, la alleno e ottengo una predizione
logReg = LogisticRegression()
logReg.fit(X_train_std, Y_train)
Y_pred = logReg.predict(X_test_std)

# eseguo una predizione anche delle confidenze delle classificazioni
# mi servirà per il calcolo della negative log-likelihood
Y_pred_proba = logReg.predict_proba(X_test_std)

# creo una matrice di confusione per analizzare il comportamento della predizione
cnf_matrix = confusion_matrix(Y_test, Y_pred)

# visualizzo la matrice di confusione in modo grafico
class_names = [0, 1]
_, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matrice di confusione', y=1.1)
plt.ylabel('Classi reali')
plt.xlabel('Classi predette')
plt.show()

# valutiamo il modello con le metriche messe a disposizione da SciKitLearn
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("Neg. Log-Likelihood:", log_loss(Y_test, Y_pred_proba))
