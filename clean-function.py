import pandas as pd
import numpy as np
import os

def limpiar_dataset(csv_file_path, output_folder="cleanDatasets"):
    # Cargar el dataset desde un archivo CSV (local o remoto)
    df = pd.read_csv(csv_file_path)
    
    # Inspección inicial
    print("Primeras filas del dataset:")
    print(df.head())

    # Inspección general de los datos
    print("\nInformación general del DataFrame:")
    print(df.info())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    # Limpiar los datos
    print("\nEliminando duplicados y rellenando valores nulos...")
    # Eliminamos filas duplicadas si existen
    df = df.drop_duplicates()

    # Rellenamos los valores nulos con la media (solo en columnas numéricas)
    df = df.fillna(df.mean(numeric_only=True))

    # Verificación final
    print("\nVista previa tras limpieza y transformación:")
    print(df.head())
    
    print("\nValores nulos después de limpieza:")
    print(df.isnull().sum())

    # Crear la carpeta para guardar los datasets limpios si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Crear un nombre para el archivo limpio usando el nombre original
    output_filename = os.path.basename(csv_file_path).replace('.csv', '_limpio.csv')

    # Guardar el dataset limpio
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path, index=False)
    print(f"\n✔️ Dataset limpio guardado como '{output_path}'")

# Limpiar y guardar varios datasets
limpiar_dataset("data/winequality-red.csv")
limpiar_dataset("data/winequality-white.csv")
