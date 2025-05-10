from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib 

def model(df):

    #MODELO
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalise: each column has mean 0 and standard deviation 1, so that the model can learn more efficiently and stably
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    #sequential is a type of network where each layer goes directly after the previous one (ideal for simple architectures)
    model = Sequential([
        #Dense creates a dense layer, it means its fully conected (each neuron is connected with all of the previous layer)
        #Dense(number of neurons, each neuron uses the activation function ReLU (Rectified Linear Unit), which introduces
        #  no-linearity, helpful for complex patterns)
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # just one output
    ])

    # configures how model is gonna be trained
    model.compile(optimizer='adam',# adam is a very famous algorithm of weights optimization
     loss='mean_squared_error',# This is the loss function(the measure the model tries to minimize),
                               # MSE isIdeal for continuos regression problems like this 
      metrics=['mae'])#set metrics used for evaluating the training, mean absolute error is easy to interpret


    #train
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)


    #evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"MAE:{mae:.2f} This tells you, on average, the model is predicting the wine quality within {mae:.2f} points on a 1–10 scale") 
    model.save('wine_quality_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
