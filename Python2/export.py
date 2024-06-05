import numpy as np  # tratamiento de datos numéricos
import matplotlib.pyplot as plt  # Sublibrería para representar
import pandas as pd
from sklearn.linear_model import LinearRegression  # Cargar y manipular datos

# Leer el dataset
dataset = pd.read_csv("50_Startups.csv")

# Seleccionar la variable independiente (todas las columnas excepto la última)
X = dataset.iloc[:, :-1].values
# Seleccionar la variable dependiente (la última columna)
y = dataset.iloc[:, 4].values

# Codificar la columna de estado (categoría) en valores numéricos
from sklearn import preprocessing
labelencoder_X = preprocessing.LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Aplicar OneHotEncoding a la columna codificada
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)

# Evitar la trampa de variables ficticias (dummy variable trap)
X = X[:, 1:]

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el modelo de regresión lineal a los datos de entrenamiento
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir los resultados en el conjunto de prueba
y_pred_test = regression.predict(X_test)
y_pred_train = regression.predict(X_train)

# Crear un DataFrame para comparar los resultados
train_results = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train, 'Set': 'Train'})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test, 'Set': 'Test'})

# Combinar los resultados de entrenamiento y prueba
results = pd.concat([train_results, test_results])

# Exportar los resultados a un archivo CSV
results.to_csv('comparison_results.csv', index=False)

print("Los resultados se han exportado correctamente a 'comparison_results.csv'")
