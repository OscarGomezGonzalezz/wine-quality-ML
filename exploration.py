import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# URL del dataset
url = "data/winequality-red.csv"

# Cargar el dataset desde la URL
df = pd.read_csv(url, sep=";")

print(df.groupby('quality').mean())


print(df.groupby('quality').mean())

# Estadísticas por calidad
print("\n🔹 Mean values grouped by wine quality:")
print(df.groupby("quality").mean())
# Calcular correlaciones
# Calcular correlaciones con la calidad
correlation = df.corr(numeric_only=True)['quality'].drop('quality').sort_values()

# Crear gráfico de barras ordenado
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation.values, y=correlation.index, palette="coolwarm")
plt.title("Correlación de cada variable con la calidad del vino")
plt.xlabel("Correlación")
plt.ylabel("Variable")
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

# top_positive = correlation.sort_values(ascending=False).head(3).index
# top_negative = correlation.sort_values().head(3).index

# # Filtrar DataFrame
# selected_features = list(top_positive) + list(top_negative) + ['quality']
# sns.pairplot(df[selected_features], hue='quality', palette="viridis", corner=True)
# plt.suptitle("Pairplot de las variables más influyentes", y=1.02)
# plt.show()


#------------------------------------------------PARTE JAVI ----------------------------------------------

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

