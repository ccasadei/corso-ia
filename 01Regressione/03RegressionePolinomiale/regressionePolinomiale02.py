import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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

# associamo ad X i valori di tutte le colonne meno MEDV
# associamo ad Y i valori di output
X = dataset.drop("MEDV", axis=1).values
Y = dataset["MEDV"].values

# suddividiamo il dataset in due dataset, uno di training ed uno di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# genero le feature per i set di training e di test con grado 3 cha abbiamo visto essere in overfitting
polyfeatures = PolynomialFeatures(degree=3)
X_train_poly = polyfeatures.fit_transform(X_train)
X_test_poly = polyfeatures.transform(X_test)

# standardizzo le feature polinomiali dei due set
ss = StandardScaler()
X_train_poly_std = ss.fit_transform(X_train_poly)
X_test_poly_std = ss.transform(X_test_poly)

# eseguo la regressione lineare multipla,
lRegr = LinearRegression()
lRegr.fit(X_train_poly_std, Y_train)
Y_pred_train = lRegr.predict(X_train_poly_std)
Y_pred_test = lRegr.predict(X_test_poly_std)

# visualizzo i risultati
mse_train = mean_squared_error(Y_train, Y_pred_train)
r2_train = r2_score(Y_train, Y_pred_train)
mse_test = mean_squared_error(Y_test, Y_pred_test)
r2_test = r2_score(Y_test, Y_pred_test)
print("Modello in overfitting già al grado 3")
print("TRAIN --- MSE:", mse_train, " - R2:", r2_train)
print("TEST ---- MSE:", mse_test, " - R2:", r2_test)
print()
print()

# preparo un array con i vari valori di lambda che vogliamo provare
# NOTA: "alpha is the new lambda"... almeno in SciKitLearn...
alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10.]

print("Usiamo la regolarizzazione L1 con Lasso")
for alpha in alphas:
    # uso il modello Lasso che utilizza la regolarizzazione L1
    model = Lasso(alpha=alpha)
    model.fit(X_train_poly_std, Y_train)

    # faccio la predizione sia su train che su test, per confrontare i risultati
    # e capire come si evolve l'overfitting
    Y_pred_train = model.predict(X_train_poly_std)
    Y_pred_test = model.predict(X_test_poly_std)

    # visualizzo i risultati
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Alpha:", alpha)
    print("TRAIN --- MSE:", mse_train, " - R2:", r2_train)
    print("TEST ---- MSE:", mse_test, " - R2:", r2_test)
    print()

print()
print("Usiamo la regolarizzazione L2 con Ridge")
for alpha in alphas:
    # uso il modello Ridge che utilizza la regolarizzazione L2
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly_std, Y_train)

    # faccio la predizione sia su train che su test, per confrontare i risultati
    # e capire come si evolve l'overfitting
    Y_pred_train = model.predict(X_train_poly_std)
    Y_pred_test = model.predict(X_test_poly_std)

    # visualizzo i risultati
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Alpha:", alpha)
    print("TRAIN --- MSE:", mse_train, " - R2:", r2_train)
    print("TEST ---- MSE:", mse_test, " - R2:", r2_test)
    print()

print()
print("Usiamo entrambe le regolarizzazioni con ElasticNet")
for alpha in alphas:
    # uso il modello ElasticNet che utilizza entrambe le regolarizzazioni
    # NOTA: il parametro l1_ratio indica a quale regolarizzazione dare più importanza
    # 0.5 -> sia L1 che L2 sono utilizzate con lo stesso peso nella regolarizzazione complessiva
    # >0.5 -> L1 ha un peso maggiore nella regolarizzazione complessiva
    # <0.5 -> L1 ha un peso minore nella regolarizzazione complessiva
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train_poly_std, Y_train)

    # faccio la predizione sia su train che su test, per confrontare i risultati
    # e capire come si evolve l'overfitting
    Y_pred_train = model.predict(X_train_poly_std)
    Y_pred_test = model.predict(X_test_poly_std)

    # visualizzo i risultati
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Alpha:", alpha)
    print("TRAIN --- MSE:", mse_train, " - R2:", r2_train)
    print("TEST ---- MSE:", mse_test, " - R2:", r2_test)
    print()
