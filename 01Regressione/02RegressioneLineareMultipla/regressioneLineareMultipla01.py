import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

'''
 Author: Cristiano Casadei
'''

# importiamo il dataset direttamente dalla URL dove è archiviato
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                      # indichiamo che il file utilizza un numero indefinito di spazi come separatore di colonna
                      sep="\s+",
                      # assegniamo i nomi alle colonne (questa volte le utilizziamo tutte!)
                      # utilizzo la nomenclatura suggerita dallo stesso fornitore
                      # che si trova a questo indirizzo
                      # https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names
                      #
                      #  0. CRIM     per capita crime rate by town
                      #  1. ZN       proportion of residential land zoned for lots over  25,000 sq.ft.
                      #  2. INDUS    proportion of non-retail business acres per town
                      #  3. CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
                      #  4. NOX      nitric oxides concentration (parts per 10 million)
                      #  5. RM       average number of rooms per dwelling
                      #  6. AGE      proportion of owner-occupied units built prior to 1940
                      #  7. DIS      weighted distances to five Boston employment centres
                      #  8. RAD      index of accessibility to radial highways
                      #  9. TAX      full-value property-tax rate per $10,000
                      # 10. PTRATIO  pupil-teacher ratio by town
                      # 11. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
                      # 12. LSTAT    % lower status of the population
                      # 13. MEDV     Median value of owner-occupied homes in $1000's
                      names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"])

# diamo un'occhiata al dataset (solo le prime righe)
print("Diamo uno sguardo al dataset...")
print(dataset.head())
print()

# diamo uno sguardo ai tipi dati
print("Diamo uno sguardo ai tipi dato...")
print(dataset.info())
print()

# valutiamo il grado di correlazione tra le proprietà del dataset
print("Grado di correlazione")
print(dataset.corr())
print()

# creiamo una heatmap delle correlazioni con seaborn
# i parametri xticklabels e yticklabels servono a fornire il nome di righe e colonne
# forniremo per entrambe l'elenco dei nomi colonna del dataset
sns.heatmap(dataset.corr(), xticklabels=dataset.columns, yticklabels=dataset.columns)
plt.show()

# creiamo i grafici delle combinazioni di coppie di proprietà
# NOTA: visto che il grafico diventerebbe enorme, selezioniamo solo le colonne che reputiamo
# più promettenti dopo aver analizzato la heatmap precedente, oltre al nostro valore di uscita
sns.pairplot(dataset[["RM", "LSTAT", "PRATIO", "TAX", "INDUS", "MEDV"]])
plt.show()

# associamo ad X i valori di input delle colonne che reputiamo più promettenti
# associamo ad Y i valori di output
X = dataset[["RM", "LSTAT"]].values
Y = dataset["MEDV"].values

# suddividiamo il dataset in due dataset, uno di training ed uno di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# istanziamo la classe di calcolo della regressione lineare di SciKitLearn
# la addestriamo e prediciamo i valori con il set di test
lRegr = LinearRegression()
lRegr.fit(X_train, Y_train)
Y_pred = lRegr.predict(X_test)

# calcoliamo l'errore quadratico medio e il coefficiente di determinazione
errore = mean_squared_error(Y_test, Y_pred)
print("Errore:", errore)
punteggio = r2_score(Y_test, Y_pred)
print("Score:", punteggio)
print()

# visualizziamo i valori dei pesi e del bias trovati
print("Valore del peso di RM:", lRegr.coef_[0])
print("Valore del peso di LSTAT:", lRegr.coef_[1])
print("Valore del bias:", lRegr.intercept_)
print()

# Ok, ora applichiamo in ingresso TUTTE le proprietà del dataset, tranne quella di output che andrà in Y
X = dataset.drop("MEDV", axis=1).values
Y = dataset["MEDV"].values

# suddividiamo il dataset in due dataset, uno di training ed uno di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# istanzio la classe di standardizzazione
# standardizzo il train set, creando un modello di standardizzazione in base ai suoi dati
# riutilizzo lo stesso modello di standardizzazione sul test set, in modo da mantenere uniformità tra i due set
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# istanziamo la classe di calcolo della regressione lineare di SciKitLearn
# la addestriamo e prediciamo i valori con il set di test
lRegr = LinearRegression()
lRegr.fit(X_train_std, Y_train)
Y_pred = lRegr.predict(X_test_std)

# calcoliamo l'errore quadratico medio e il coefficiente di determinazione
errore = mean_squared_error(Y_test, Y_pred)
print("Errore:", errore)
punteggio = r2_score(Y_test, Y_pred)
print("Score:", punteggio)
print()

# visualizziamo i valori dei pesi (con i rispettivi nomi delle colonne) e del bias trovati
print("Valore dei pesi:", list(zip(dataset.columns, lRegr.coef_)))
print("Valore del bias:", lRegr.intercept_)
