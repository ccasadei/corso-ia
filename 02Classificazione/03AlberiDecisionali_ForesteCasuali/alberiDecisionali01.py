import graphviz as graphviz
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

'''
 Author: Cristiano Casadei
'''

# carico il dataset
indianLiver = pd.read_csv("./liver.csv")
print("Diamo uno sguardo alla struttura dati")
print(indianLiver.info())
print()
print("Diamo uno sguardo ai dati")
print(indianLiver.head())
print()

# codifico le colonne non numeriche con delle colonne booleane dummy
# in questo caso la colonna "gender" verrà sostituita da due colonne
# "gender_female" e "gender_male" che indicheranno l'appartenenza al genere
indianLiver = pd.get_dummies(indianLiver)
print("Il dataset modificato con le colonne dummies del sesso")
print(indianLiver.head())
print()

# creo i dataset delle proprietà X e del target Y
X = indianLiver.drop("LABEL", axis=1).values
Y = indianLiver["LABEL"].values

# elimino le righe dove compare un NaN nei valori (darebbe errore in fase di elaborazione)
Y = Y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]

# suddivido il dataset in dataset di train e di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# NOTA: gli alberi decisionali non richiedono normalizzazione o standardizzazione!

# istanzio la classe di creazione di alberi decisionali, utilizzando l'algoritmo Gini per calcolare l'impurità
decTree = DecisionTreeClassifier(criterion="gini")
decTree.fit(X_train, Y_train)

# calcolo le predizioni su train set e test set
Y_pred_train = decTree.predict(X_train)
Y_pred_test = decTree.predict(X_test)

# calcolo l'accuracy
accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test, Y_pred_test)

print("Uso l'algoritmo 'gini' per l'impurità")
print("Profondità albero:", decTree.tree_.max_depth)
print("Accuracy train set:", accuracy_train)
print("Accuracy test set:", accuracy_test)
print()

# metto in grafica l'albero decisionale e creo un pdf
dot_data = export_graphviz(decTree, out_file=None,
                           feature_names=indianLiver.columns.drop("LABEL"))
graph = graphviz.Source(dot_data)
graph.render("liver_tree_gini", view=True)

# rifaccio usando l'entropia
decTree = DecisionTreeClassifier(criterion="entropy")
decTree.fit(X_train, Y_train)

# calcolo le predizioni su train set e test set
Y_pred_train = decTree.predict(X_train)
Y_pred_test = decTree.predict(X_test)

# calcolo l'accuracy
accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test, Y_pred_test)

print("Uso l'algoritmo 'entropy' per l'impurità")
print("Profondità albero:", decTree.tree_.max_depth)
print("Accuracy train set:", accuracy_train)
print("Accuracy test set:", accuracy_test)
print()

# metto in grafica l'albero decisionale e creo un pdf
dot_data = export_graphviz(decTree, out_file=None,
                           feature_names=indianLiver.columns.drop("LABEL"))
graph = graphviz.Source(dot_data)
graph.render("liver_tree_entropy", view=True)

# rifaccio usando nuovamente Gini, ma limitando la profondità dell'albero
decTree = DecisionTreeClassifier(criterion="gini", max_depth=2)
decTree.fit(X_train, Y_train)

# calcolo le predizioni su train set e test set
Y_pred_train = decTree.predict(X_train)
Y_pred_test = decTree.predict(X_test)

# calcolo l'accuracy
accuracy_train = accuracy_score(Y_train, Y_pred_train)
accuracy_test = accuracy_score(Y_test, Y_pred_test)

print("Uso l'algoritmo 'gini' per l'impurità e limito a 2 la profondità dell'albero")
print("Profondità albero:", decTree.tree_.max_depth)
print("Accuracy train set:", accuracy_train)
print("Accuracy test set:", accuracy_test)
print()

# metto in grafica l'albero decisionale e creo un pdf
dot_data = export_graphviz(decTree, out_file=None,
                           feature_names=indianLiver.columns.drop("LABEL"))
graph = graphviz.Source(dot_data)
graph.render("liver_tree_gini_depth2", view=True)
