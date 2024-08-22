import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import joblib

# Ruta del archivo CSV para almacenar datos
data_log_file = 'trading_data_log.csv'

# Leer los datos del archivo CSV
data = pd.read_csv(data_log_file)

# Filtrar solo las entradas correctas
data = data[data['Result'] == 'Correcto']

# Si no hay suficientes datos, no realizar el reentrenamiento
if len(data) < 10:
    print("No hay suficientes datos correctos para reentrenar el modelo.")
    exit()

# Preprocesar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

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

print("Creando conjuntos de datos...")
time_step = 2  # Ajusta el número de pasos de tiempo según sea necesario
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Cargar el modelo existente
print("Cargando el modelo LSTM existente...")
model = load_model('lstm_model.h5')

# Compilar el modelo
print("Compilando el modelo...")
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con los nuevos datos
print("Reentrenando el modelo...")
model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data=(X_test, y_test))

# Guardar el modelo y el scaler actualizados
model.save('lstm_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo reentrenado y guardado con éxito.")
