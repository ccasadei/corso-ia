import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

'''
 Author: Cristiano Casadei
'''

# importiamo il solito dataset al completo
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                      sep="\s+",
                      names=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PRATIO", "B", "LSTAT", "MEDV"])

print("Utilizzo solo LSTAT")
print()
# associamo ad X i valori di input di LSTAT
# associamo ad Y i valori di output
X = dataset[["LSTAT"]].values
Y = dataset["MEDV"].values

# suddividiamo il dataset in due dataset, uno di training ed uno di test
# questa volta uso un random state fisso, in modo che i dati vengano mischiati, ma sempre allo stesso modo
# così da poter confrontare direttamente i vari metodi di regressione
# in pratica random_state è il seed del generatore random
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# ciclo il procedimento per vedere i risultati dei vari gradi del polinomio
# in questo caso vado da grado 1 (lineare) a grado 10
for grado in range(1, 11):
    # istanzo il generatore di feature polinomiali per il grado corrente
    # successivamente lo utilizzo per generare le feature per i set di training e di test
    polyfeatures = PolynomialFeatures(degree=grado)
    X_train_poly = polyfeatures.fit_transform(X_train)
    X_test_poly = polyfeatures.transform(X_test)

    # istanzio la classe di standardizzazione e standardizzo le feature polinomiali dei due set
    ss = StandardScaler()
    X_train_poly_std = ss.fit_transform(X_train_poly)
    X_test_poly_std = ss.transform(X_test_poly)

    # la regressione polinomiale è un caso particolare di regressione lineare multipla,
    # quindi proseguo come già visto utilizzando però i set arricchiti con le feature polinomiali
    lRegr = LinearRegression()
    lRegr.fit(X_train_poly_std, Y_train)
    Y_pred = lRegr.predict(X_test_poly_std)

    # calcoliamo l'errore quadratico medio e il coefficiente di determinazione
    errore = mean_squared_error(Y_test, Y_pred)
    punteggio = r2_score(Y_test, Y_pred)
    print("Grado:", grado, " - MSE:", punteggio, " - R2:", punteggio)

print()
print()
print("Utilizzo tutte le colonne")
print()
# associamo ad X i valori di tutte le colonne meno MEDV
# associamo ad Y i valori di output
X = dataset.drop("MEDV", axis=1).values
Y = dataset["MEDV"].values

# suddividiamo il dataset in due dataset, uno di training ed uno di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# ciclo da grado 1 (lineare) a grado 4
for grado in range(1, 5):
    # genero le feature per i set di training e di test
    polyfeatures = PolynomialFeatures(degree=grado)
    X_train_poly = polyfeatures.fit_transform(X_train)
    X_test_poly = polyfeatures.transform(X_test)

    # standardizzo le feature polinomiali dei due set
    ss = StandardScaler()
    X_train_poly_std = ss.fit_transform(X_train_poly)
    X_test_poly_std = ss.transform(X_test_poly)

    # eseguo la regressione lineare multipla,
    lRegr = LinearRegression()
    lRegr.fit(X_train_poly_std, Y_train)
    Y_pred = lRegr.predict(X_test_poly_std)

    # visualizzo i risultati
    errore = mean_squared_error(Y_test, Y_pred)
    punteggio = r2_score(Y_test, Y_pred)
    print("Grado:", grado, " - MSE:", punteggio, " - R2:", punteggio)

print()
print()
