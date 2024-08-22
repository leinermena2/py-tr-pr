import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np
import pandas as pd
import time
import re
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Configurar la ruta de Tesseract si es necesario (solo para Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Coordenadas del área donde se encuentra el gráfico de velas
candlestick_area_coords = (87, 160, 1652, 945)  # (x1, y1, x2, y2)

# Coordenadas del área donde se encuentran los valores open, high, low, close
price_area_coords = (139, 880, 236, 955)  # (x1, y1, x2, y2) 

# Función para mejorar la calidad de la imagen capturada
def improve_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=3.0, beta=0)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return sharpened

# Preprocesamiento de la imagen para encontrar contornos de velas
def preprocess_image_for_candles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    _, thresh = cv2.threshold(adjusted, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Detectar contornos y características de velas japonesas
def detect_candles(image):
    processed_image = preprocess_image_for_candles(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and w < 20:  # Filtrar posibles velas según tamaño (ajusta los valores según sea necesario)
            candles.append((x, y, w, h))
    
    return candles

# Validar si el texto extraído tiene un formato de número decimal correcto
def is_valid_price(price_str):
    try:
        float(price_str)
        return True
    except ValueError:
        return False

# Extraer precios desde la imagen utilizando OCR
def extract_prices_from_image():
    screen_price = np.array(ImageGrab.grab(bbox=price_area_coords))
    processed_region = improve_image_quality(screen_price)
    
    text = pytesseract.image_to_string(processed_region)
    print("Texto extraído:", text)
    
    prices = re.findall(r"\d+\.\d+", text)
    
    # Validar y convertir precios a flotantes
    valid_prices = [float(price) for price in prices if is_valid_price(price)]
    
    if len(valid_prices) >= 4:
        open_price = valid_prices[0]
        high_price = valid_prices[1]
        low_price = valid_prices[2]
        close_price = valid_prices[3]
        print(f"Precios extraídos: Open={open_price}, High={high_price}, Low={low_price}, Close={close_price}")
        return open_price, high_price, low_price, close_price
    
    print("No se pudieron extraer precios suficientes. Reintentando...")
    return None, None, None, None

# Capturar y extraer datos de precios y características de velas, incluyendo la moneda seleccionada
def capture_and_extract_data(moneda, duration_hours):
    data = []
    end_time = time.time() + duration_hours * 3600
    
    while time.time() < end_time:  # Captura de datos durante la duración especificada
        try:
            print(f"Capturando muestra para {moneda}...")

            open_price, high_price, low_price, close_price = extract_prices_from_image()
            if open_price is not None:
                # Codificación one-hot para la moneda seleccionada
                monedas = [
                    'AUD/CAD', 'AUD/CHF', 'CAD/CHF', 'AUD/CAD OTC', 
                    'AUD/CHF OTC', 'EUR/CAD OTC', 'USD/CAD OTC', 
                    'eur/usd otc', 'aud/usd otc', 'EUROPE COMPOSITE INDEX', 
                    'EUR/AUD', 'EUT/CAD'
                ]
                moneda_feature = [1 if m == moneda else 0 for m in monedas]

                data.append([open_price, high_price, low_price, close_price] + moneda_feature)
            else:
                print("No se pudieron extraer los precios de la imagen. Reintentando...")
                time.sleep(1)
                continue

            screen_candles = np.array(ImageGrab.grab(bbox=candlestick_area_coords))
            candles = detect_candles(screen_candles)
            if candles:
                print(f"Se detectaron {len(candles)} velas.")
            else:
                print("No se detectaron velas.")

            time.sleep(1)  # Espera 1 segundo entre capturas

            if len(data) >= 3600:  # Si has capturado suficientes datos, continúa con el procesamiento
                columns = ['Open', 'High', 'Low', 'Close'] + monedas
                df = pd.DataFrame(data, columns=columns)
                return df

        except KeyboardInterrupt:
            print("Captura de datos interrumpida manualmente.")
            break

# Función para registrar las horas de entrenamiento acumuladas
def record_training_time(moneda, duration_hours):
    time_file = f'training_time_{moneda.replace("/", "_").replace(" ", "_")}.txt'
    if os.path.exists(time_file):
        with open(time_file, 'r') as f:
            total_hours = float(f.read())
    else:
        total_hours = 0.0

    total_hours += duration_hours

    with open(time_file, 'w') as f:
        f.write(str(total_hours))

    print(f"Total de horas de entrenamiento acumuladas para {moneda}: {total_hours}")

# Función para mostrar las activaciones de las capas del modelo
def visualize_layer_activations(model, data_sample):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(data_sample)

    for layer_activation in activations:
        plt.matshow(layer_activation[0, :, :], cmap='viridis')
        plt.show()

# Función para entrenar el modelo, con soporte para múltiples monedas y aprendizaje continuo
def train_model_continuously():
    monedas = [
        'AUD/CAD', 'AUD/CHF', 'CAD/CHF', 'AUD/CAD OTC', 
        'AUD/CHF OTC', 'EUR/CAD OTC', 'USD/CAD OTC', 
        'eur/usd otc', 'aud/usd otc', 'EUROPE COMPOSITE INDEX', 
        'EUR/AUD', 'EUT/CAD'
    ]

    # Mostrar lista de monedas para seleccionar
    print("Seleccione la moneda para el entrenamiento:")
    for idx, moneda in enumerate(monedas, 1):
        print(f"{idx}. {moneda}")
    
    selected_index = int(input("Ingrese el número de la moneda: ")) - 1
    moneda = monedas[selected_index]

    model_file = f'lstm_model_{moneda.replace("/", "_").replace(" ", "_")}.h5'
    scaler_file = f'scaler_{moneda.replace("/", "_").replace(" ", "_")}.pkl'

    if os.path.exists(model_file):
        print(f"Cargando modelo existente para {moneda}...")
        model = load_model(model_file)
        model.compile(optimizer='adam', loss='mean_squared_error')
    else:
        print(f"Creando un nuevo modelo LSTM para {moneda}...")
        model = Sequential()
        model.add(LSTM(150, return_sequences=True, input_shape=(20, 16), kernel_regularizer=l2(0.01)))  # Regularización L2
        model.add(LSTM(150, return_sequences=False))
        model.add(Dropout(0.5))  # Aumentar Dropout para regularización
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))

    # Configuración de callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    log_dir = f"logs/fit/{moneda}_{time.strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    try:
        duration_hours = float(input("Ingrese la duración del entrenamiento en horas: "))
        
        while True:
            print(f"Iniciando nueva captura de datos para {moneda}...")
            df = capture_and_extract_data(moneda, duration_hours)

            if df is not None and not df.empty:
                print("Preprocesando datos...")
                scaled_data = scaler.fit_transform(df)
                joblib.dump(scaler, scaler_file)

                train_size = int(len(scaled_data) * 0.8)
                test_size = len(scaled_data) - train_size
                train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

                def create_dataset(dataset, time_step=20):
                    X, Y = [], []
                    for i in range(len(dataset) - time_step):
                        X.append(dataset[i:(i + time_step), :])
                        Y.append(dataset[i + time_step, 3])
                    return np.array(X), np.array(Y)

                print("Creando conjuntos de datos...")
                X_train, y_train = create_dataset(train_data, 20)
                X_test, y_test = create_dataset(test_data, 20)

                print("Entrenando el modelo...")
                model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1, 
                          validation_data=(X_test, y_test), 
                          callbacks=[early_stopping, reduce_lr, tensorboard_callback])

                # Evaluación del modelo
                loss = model.evaluate(X_test, y_test)
                print(f"Loss en conjunto de prueba: {loss}")

                # Visualización de activaciones de capas
                print("Visualizando activaciones de capas para una muestra de datos...")
                visualize_layer_activations(model, X_test[:1])

                # Generar predicciones en los datos de prueba
                y_pred = model.predict(X_test)
                y_pred_rescaled = scaler.inverse_transform(
                    np.concatenate([np.zeros((len(y_pred), len(df.columns) - 1)), y_pred], axis=1)
                )[:, -1]
                # Comparar predicciones con los valores reales
                plt.figure(figsize=(14, 7))
                plt.plot(y_test, color='blue', label='Real Price')
                plt.plot(y_pred_rescaled, color='red', label='Predicted Price')
                plt.title(f'Predicciones vs Precios Reales para {moneda}')
                plt.xlabel('Tiempo')
                plt.ylabel('Precio')
                plt.legend()
                plt.show()

                # Guardar el modelo entrenado
                model.save(model_file)
                print(f"Modelo guardado exitosamente como {model_file}.")
            else:
                print("No se han capturado suficientes datos para entrenar el modelo.")

            # Registrar las horas de entrenamiento acumuladas
            record_training_time(moneda, duration_hours)

            break  # Salir del bucle después de la sesión de entrenamiento

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido manualmente.")

# Ejecutar el entrenamiento continuo
train_model_continuously()
