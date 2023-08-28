# Regresión Lineal Múltiple
'''Pandas trae una función para hacer la dummificación
Cuando se carga la X no pongas al final ".values" basta hacer "get_dummies"

X = df.iloc[:,:-1]
X = pd.get_dummies(X, columns=['State']).values

Otra opcion:
dummies = pandas.get_dummies(x[:, 3]).values
x = numpy.concatenate([x[:, :-1], dummies[:, :-1]], axis=1)'''

import numpy as np
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])#el dato categorico se encuentra en la columna 3
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Evitar la trampa de las variables ficticias
X = X[:, 1:]#Se elimina una de las variables Dummy

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de Regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)


# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)#agregar una columna
SL = 0.05

#se tiene que volver a calcular el modelo con las variables iniciales
X_opt = X[:, [0, 1, 2, 3, 4, 5]]#sellecciona todas las columanas(variables)
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()#endog=variable a predecir/exog=variable predictoras
regression_OLS.summary()#para observar el p-valor

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()
'''tomando en cuenta un criterio de eliminacion se pueden 
eliminar mas o menor cantidad de variables'''

#ajuste del modelo automatico
'''    
import statsmodels.formula.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
 '''