import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np
import pandas as pd
from datetime import datetime
import time
import re
import pyautogui
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Configurar la ruta de Tesseract si es necesario (solo para Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Coordenadas del área donde se encuentran los precios en la pantalla
price_area_coords = (141, 877, 233, 956)  # (x1, y1, x2, y2)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def extract_prices_from_image(image):
    # Extraer la región específica de la imagen que contiene los precios
    x1, y1, x2, y2 = price_area_coords
    price_region = image[y1:y2, x1:x2]
    
    # Convertir la imagen a texto usando Tesseract OCR
    text = pytesseract.image_to_string(price_region)
    print("Texto extraído:", text)  # Imprimir el texto extraído para depuración
    
    # Usar expresiones regulares para encontrar los valores de precios
    open_price = re.search(r"(open|opne|oen|ope|oper|pen)\s+(\d+\.\d+)", text, re.IGNORECASE)
    high_price = re.search(r"(high|hgh|hig|hihg|igh|hig|figh|igh)\s+(\d+\.\d+)", text, re.IGNORECASE)
    low_price = re.search(r"(low|lwo|lw|lo|ow|tow|tow)\s+(\d+\.\d+)", text, re.IGNORECASE)
    close_price = re.search(r"(close|clsoe|cls|cloe|dose|doce|cose|lose)\s+(\d+\.\d+)", text, re.IGNORECASE)
    
    if open_price and high_price and low_price and close_price:
        return float(open_price.group(2)), float(high_price.group(2)), float(low_price.group(2)), float(close_price.group(2))
    return None, None, None, None

def capture_and_extract_data(num_samples=190):
    data = []
    for i in range(num_samples):
        print(f"Capturando muestra {i+1}/{num_samples}")
        screen = np.array(ImageGrab.grab())
        
        graph_region = preprocess_image(screen)
        open_price, high_price, low_price, close_price = extract_prices_from_image(graph_region)
        if open_price is not None:
            data.append([open_price, high_price, low_price, close_price])
        else:
            print("No se pudieron extraer los precios de la imagen.")
        time.sleep(2)  # Espera 2 segundos entre capturas

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    return df

# Capturar y extraer datos de precios
print("Iniciando captura de datos...")
df = capture_and_extract_data(num_samples=190)
print(df)

# Preprocesamiento de los datos
print("Preprocesando datos...")
if not df.empty:
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

    print("Creando conjuntos de datos...")
    time_step = 2  # Ajusta el número de pasos de tiempo según sea necesario
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Crear el modelo LSTM
    print("Creando el modelo LSTM...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, scaled_data.shape[1])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compilar el modelo
    print("Compilando el modelo...")
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    print("Entrenando el modelo...")
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    def make_prediction(features):
        features = np.array(features).reshape(1, -1, scaled_data.shape[1])
        scaled_features = scaler.transform(features[0]).reshape(1, -1, scaled_data.shape[1])
        prediction = model.predict(scaled_features)
        return scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

    def automate_click(decision):
        try:
            if decision == "Subir":
                pyautogui.click(x=1171, y=538)
            elif decision == "Bajar":
                pyautogui.click(x=1813, y=484)
        except Exception as e:
            print(f"Error en la automatización de clic: {e}")

    def capture_and_predict():
        while True:
            try:
                print("Capturando pantalla para predicción...")
                screen = np.array(ImageGrab.grab())
                graph_region = preprocess_image(screen)
                open_price, high_price, low_price, close_price = extract_prices_from_image(graph_region)
                if open_price is not None:
                    features = [open_price, high_price, low_price, close_price]
                    prediction = make_prediction([features])
                    print(f'Predicción del precio de cierre: {prediction}')
                    
                    # Aquí se puede decidir si hacer clic en "Subir" o "Bajar" basado en la predicción
                    if prediction > close_price:
                        automate_click("Subir")
                    else:
                        automate_click("Bajar")

                time.sleep(120)  # Espera 2 minutos entre predicciones
            except Exception as e:
                print(f"Error en el bucle principal: {e}")

    # Iniciar la captura y predicción en tiempo real
    print("Iniciando captura y predicción en tiempo real...")
    capture_and_predict()
else:
    print("No se han capturado suficientes datos para entrenar el modelo.")
