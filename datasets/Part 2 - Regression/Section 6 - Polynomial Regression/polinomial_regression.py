# Regresión polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''Con un conjunto de datos pequeño no es necesario dividir 
en conjunto de entrenamiento y testing'''

# Ajusta la regresión Lineal con el dataset (no es la adecuada)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresión POLINÓMICA con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)#grado del polinomio
X_poly = poly_reg.fit_transform(X)#se aumentan los polinomios en la matriz(trasformacion)
lin_reg_2 = LinearRegression()#la misma para hacer una regresion Lineal
lin_reg_2.fit(X_poly, y)

# Visualización de los resultados del Modelo Lineal
# plt.figure()
# plt.scatter(X, y, color = "red")#puntos originales
# plt.plot(X, lin_reg.predict(X), color = "blue")
# plt.title("Modelo de Regresión Lineal")
# plt.xlabel("Posición del empleado")
# plt.ylabel("Sueldo (en $)")
# plt.show()

# Visualización de los resultados del Modelo Polinómico
plt.figure()
X_grid = np.arange(min(X), max(X), 0.1)#Para "suavisar la linea"
X_grid = X_grid.reshape(len(X_grid), 1)#pasar a columna

plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Predicción del modelo
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))






