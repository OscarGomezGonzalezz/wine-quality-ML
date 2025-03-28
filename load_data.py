import pandas as pd

# URL del dataset
url = "data/winequality-red.csv"

# Cargar el dataset desde la URL
df = pd.read_csv(url, sep=";")

# Ver las primeras filas
print(df.head())
