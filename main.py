from clean import clean_dataset 
from exploration import explore
from model import model
from tensorflow.keras.models import load_model
import joblib 
import numpy as np



# explore(df1)
# explore(df2)


model(df1)

wine_model = load_model('wine_quality_model.h5')
scaler = joblib.load('scaler.pkl')


sample = df1.drop('quality', axis=1).sample(1, random_state=42)

# Escalamos la fila como se hizo en el entrenamiento
sample_scaled = scaler.transform(sample)

# Predecimos la calidad
predicted_quality = wine_model.predict(sample_scaled)


print(f"\nInput data:\n{sample}")
print(f"\nPredicted wine quality (1–10 scale): {predicted_quality[0][0]:.2f}")

