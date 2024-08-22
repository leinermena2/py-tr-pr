import pandas as pd
import numpy as np
from keras.models import load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
import cv2
import pytesseract
from PIL import ImageGrab
import re
import time
from datetime import datetime

# Configurar la ruta de Tesseract si es necesario (solo para Windows)
pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
    high_price = re.search(r"(high|hgh|hig|hihg|igh|hig|figh|igh|hugh|ugh|fagh|agh)\s+(\d+\.\d+)", text, re.IGNORECASE)
    low_price = re.search(r"(low|lwo|lw|lo|ow|tow|tow)\s+(\d+\.\d+)", text, re.IGNORECASE)
    close_price = re.search(r"(close|clsoe|cls|cloe|dose|doce|cose|lose|dow|dom)\s+(\d+\.\d+)", text, re.IGNORECASE)
    
    if open_price and high_price and low_price and close_price:
        return float(open_price.group(2)), float(high_price.group(2)), float(low_price.group(2)), float(close_price.group(2))
    return None, None, None, None

def capture_and_extract_data(num_samples=500):
    data = []
    for i in range(num_samples):
        print(f"Capturando muestra {i+1}/{num_samples}")
        screen = np.array(ImageGrab.grab())
        
        graph_region = preprocess_image(screen)
        open_price, high_price, low_price, close_price = extract_prices_from_image(graph_region)
        if open_price is not None:
            # Capturar la hora actual
            current_time = datetime.now().hour + datetime.now().minute / 60.0
            data.append([open_price, high_price, low_price, close_price, current_time])
        else:
            print("No se pudieron extraer los precios de la imagen.")
        time.sleep(2)  # Espera 2 segundos entre capturas

    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close', 'Time'])
    return df

# Cargar el modelo previamente entrenado y el scaler
model = load_model('lstm_model.h5')
scaler = joblib.load('scaler.pkl')

# Capturar y extraer nuevos datos de precios
print("Iniciando captura de nuevos datos...")
df_new = capture_and_extract_data(num_samples=500)  # Por ejemplo, 500 nuevas muestras
print(df_new)

# Preprocesamiento de los nuevos datos
if not df_new.empty:
    scaled_data_new = scaler.transform(df_new)

    # Crear conjuntos de datos de entrenamiento con los nuevos datos
    train_size_new = int(len(scaled_data_new) * 0.8)
    test_size_new = len(scaled_data_new) - train_size_new
    train_data_new, test_data_new = scaled_data_new[0:train_size_new, :], scaled_data_new[train_size_new:len(scaled_data_new), :]

    # Función para crear la secuencia de datos
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), :])
            Y.append(dataset[i + time_step, 3])  # Columna 'Close'
        return np.array(X), np.array(Y)

    print("Creando conjuntos de datos con nuevos datos...")
    time_step = 2  # Ajusta el número de pasos de tiempo según sea necesario
    X_train_new, y_train_new = create_dataset(train_data_new, time_step)
    X_test_new, y_test_new = create_dataset(test_data_new, time_step)

    # Continuar entrenando el modelo con los nuevos datos
    print("Reentrenando el modelo con nuevos datos...")
    model.fit(X_train_new, y_train_new, batch_size=1, epochs=1)

    # Guardar el modelo actualizado
    model.save('lstm_model_updated.h5')
else:
    print("No se han capturado suficientes datos nuevos para reentrenar el modelo.")
