import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np
import pandas as pd
import time
import re
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Configurar la ruta de Tesseract si es necesario (solo para Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Coordenadas del área donde se encuentra el gráfico de velas
candlestick_area_coords = (87, 164, 1605, 937)  # (x1, y1, x2, y2)

# Coordenadas del área donde se encuentran los valores open, high, low, close
price_area_coords = (87, 877, 233, 956)  # (x1, y1, x2, y2)

# Función para mejorar la calidad de la imagen capturada
def improve_image_quality(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumentar el contraste
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Aplicar umbralización adaptativa para mejorar el contraste local
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Aumentar la nitidez (aplicando un filtro de kernel)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened

# Preprocesamiento de la imagen para encontrar contornos de velas
def preprocess_image_for_candles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ajustar brillo y contraste para mejorar la detección de velas
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

# Extraer precios desde la imagen utilizando OCR
def extract_prices_from_image():
    # Captura de la región de precios por separado
    screen_price = np.array(ImageGrab.grab(bbox=price_area_coords))
    processed_region = improve_image_quality(screen_price)
    
    text = pytesseract.image_to_string(processed_region)
    print("Texto extraído:", text)  # Imprimir el texto extraído para depuración
    
    open_price = re.search(r"(open|opne|oen|ope|oper|pen)\s*(\d+\.\d+)", text, re.IGNORECASE)
    high_price = re.search(r"(high|hgh|hig|hihg|igh|hig|figh|igh|fagh|hgih|hugh|tagh|ugh)\s*(\d+\.\d+)", text, re.IGNORECASE)
    low_price = re.search(r"(low|lwo|lw|lo|ow|tow|tow|fow)\s*(\d+\.\d+)", text, re.IGNORECASE)
    close_price = re.search(r"(close|clsoe|cls|cloe|dose|doce|cose|lose|dove|love)\s*(\d+\.\d+)", text, re.IGNORECASE)
    
    if open_price and high_price and low_price and close_price:
        return float(open_price.group(2)), float(high_price.group(2)), float(low_price.group(2)), float(close_price.group(2))
    return None, None, None, None

# Capturar y extraer datos de precios y características de velas durante 2 minutos
def capture_and_extract_data():
    data = []
    num_samples = 24  # 2 minutos a intervalos de 5 segundos (24 muestras en 2 minutos)
    
    for i in range(num_samples):
        print(f"Capturando muestra {i+1}/{num_samples}")
        
        # Capturar velas
        screen_candles = np.array(ImageGrab.grab(bbox=candlestick_area_coords))
        candles = detect_candles(screen_candles)
        if candles:
            print(f"Se detectaron {len(candles)} velas.")
        else:
            print("No se detectaron velas.")
        
        # Captura de precios por separado
        open_price, high_price, low_price, close_price = extract_prices_from_image()
        if open_price is not None:
            data.append([open_price, high_price, low_price, close_price])
        else:
            print("No se pudieron extraer los precios de la imagen.")
        
        time.sleep(5)  # Espera 5 segundos entre capturas

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    return df

# Función para entrenar el modelo, con soporte para aprendizaje continuo
def train_model_continuously():
    # Cargar modelo existente o crear uno nuevo
    model_file = 'lstm_model.h5'
    if os.path.exists(model_file):
        print("Cargando modelo existente...")
        model = load_model(model_file)
    else:
        print("Creando un nuevo modelo LSTM...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(2, 4)))  # time_step=2, 4 features (Open, High, Low, Close)
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Escalador (normalización)
    scaler = None
    if os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))

    try:
        while True:
            print("Iniciando nueva captura de datos de 2 minutos...")
            df = capture_and_extract_data()

            if not df.empty:
                print("Preprocesando datos...")
                scaled_data = scaler.fit_transform(df)
                joblib.dump(scaler, 'scaler.pkl')  # Guardar el scaler para su uso posterior

                # Crear conjuntos de datos de entrenamiento y prueba
                train_size = int(len(scaled_data) * 0.8)
                test_size = len(scaled_data) - train_size
                train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

                # Función para crear la secuencia de datos
                def create_dataset(dataset, time_step=2):
                    X, Y = [], []
                    for i in range(len(dataset) - time_step):
                        X.append(dataset[i:(i + time_step), :])
                        Y.append(dataset[i + time_step, 3])  # Columna 'Close'
                    return np.array(X), np.array(Y)

                print("Creando conjuntos de datos...")
                time_step = 2
                X_train, y_train = create_dataset(train_data, time_step)
                X_test, y_test = create_dataset(test_data, time_step)

                # Entrenar el modelo con los nuevos datos
                print("Entrenando el modelo...")
                model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)

                # Guardar el modelo entrenado
                model.save(model_file)
                print("Modelo guardado exitosamente.")
            else:
                print("No se han capturado suficientes datos para entrenar el modelo.")

            # Esperar 2 minutos antes de la próxima iteración
            time.sleep(120)

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido manualmente.")

# Ejecutar el entrenamiento continuo
train_model_continuously()
