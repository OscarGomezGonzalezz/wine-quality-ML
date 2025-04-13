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

print(wine.groupby('quality').mean())