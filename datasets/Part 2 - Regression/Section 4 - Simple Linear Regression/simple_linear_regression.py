# Regresión Lineal Simple

import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Escalado de variables
"""CUANDO ES REGRESION LINEAL SIMPLE NO HACE FALRA ESCALADO
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""


#Modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)#deben ser del mismo tamaño

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Diferencia entre prediccion de modelo y datos reales
plt.figure('diferencia resultados')
plt.scatter(X_test, y_test, color = "red"), plt.show()
plt.scatter(X_test,y_pred, color = 'blue'), plt.show()
plt.figure()
plt.plot(y_test)
plt.plot(y_pred)

# Visualizar los resultados de entrenamiento
plt.figure()
plt.scatter(X_train, y_train, color = "red")#Grafica una nube de puntos
plt.plot(X_train, regression.predict(X_train), color = "blue")#muestra los puntos predictos en entrenamiento
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los resultados de test
plt.figure()
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, regression.predict(X_test), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

