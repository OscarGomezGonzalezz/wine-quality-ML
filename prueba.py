import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Leer el CSV
data = pd.read_csv('./data/winequality-red.csv', sep=';')

# Ver primeras filas
print(data.head())

# Información general
print(data.info())

# Resumen estadístico
print(data.describe())



sns.countplot(x='quality', data=data)
plt.title('Distribución de Calidad de Vinos')
plt.show()

corr = data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

features = data.columns[:-1]  # Todas menos 'quality'

for feature in features:
    plt.figure()
    sns.boxplot(x='quality', y=feature, data=data)
    plt.title(f'Relación entre {feature} y Calidad')
    plt.show()


#MODELO
X = data.drop('quality', axis=1)
y = data['quality']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import numpy as np

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Importancia de las Características')
plt.show()
