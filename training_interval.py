import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar los datos de entrenamiento
# Asegúrate de que los datos reflejen el intervalo de tiempo adecuado
# Aquí se asume que los datos están en un DataFrame llamado 'df' con columnas ['Open', 'High', 'Low', 'Close']

# Preprocesamiento de los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Crear conjuntos de datos de entrenamiento y prueba
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Función para crear la secuencia de datos
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 3])  # Columna 'Close'
    return np.array(X), np.array(Y)

# Definir el tiempo de paso (ej. 2 minutos)
time_step = 2
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, scaled_data.shape[1])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Guardar el modelo y el scaler
model.save('lstm_model.h5')
joblib.dump(scaler, 'scaler.pkl')
