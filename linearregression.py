import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Cargar dataset
df = pd.read_csv('./data/winequality-red.csv', sep=';', quotechar='"')  # También puedes usar 'winequality-white.csv'

# Asegurar que la calidad esté en rango [1, 10] (reescala si fuera necesario)
# En este dataset, la calidad suele ir de 3 a 8
# Vamos a reescalarla linealmente al rango [1, 10]
min_q, max_q = df['quality'].min(), df['quality'].max()
df['quality'] = df['quality'].apply(lambda x: 1 + 9 * (x - min_q) / (max_q - min_q))

# Separar variables predictoras y target
X = df.drop('quality', axis=1)
y = df['quality']

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# ✅ Forzar salida entre 1 y 10
y_pred = np.clip(y_pred, 1, 10)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Gráfico
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Calidad real")
plt.ylabel("Calidad predicha")
plt.title("Regresión lineal: calidad real vs predicha")
plt.plot([1, 10], [1, 10], '--r')
plt.grid(True)
plt.show()
