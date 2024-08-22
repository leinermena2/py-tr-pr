import cv2
import pytesseract
from PIL import ImageGrab
import numpy as np
import re
import pyautogui
import joblib
import time
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import winsound
from collections import deque
import logging

# Configurar la ruta de Tesseract si es necesario (solo para Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configurar logging para registrar eventos, errores y decisiones
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Coordenadas del área donde se encuentran los precios en la pantalla
price_area_coords = (141, 877, 233, 956)  # (x1, y1, x2, y2)

# Coordenadas a las que el cursor debe volver después de hacer clic
return_coords = (1066, 408)

# Saldo inicial y monto de apuestas
saldo = 5503
monto_apuesta = 20
ganancia_por_victoria = 17

# Configuración de sonido
sound_enabled = True  # Cambia a False si no quieres que suenen los sonidos

# Historial de resultados (últimas 50 operaciones)
result_history = deque(maxlen=50)

# Listado de monedas
monedas = [
    'AUD/CAD', 'AUD/CHF', 'CAD/CHF', 'AUD/CAD OTC', 
    'AUD/CHF OTC', 'EUR/CAD OTC', 'USD/CAD OTC', 
    'eur/usd otc', 'aud/usd otc', 'EUROPE COMPOSITE INDEX', 
    'EUR/AUD', 'EUT/CAD'
]

# Función para mejorar la calidad de la imagen capturada
def improve_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=3.0, beta=0)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return sharpened

def extract_prices_from_image():
    screen_price = np.array(ImageGrab.grab(bbox=price_area_coords))
    processed_region = improve_image_quality(screen_price)
    
    text = pytesseract.image_to_string(processed_region)
    logging.info(f'Texto extraído: {text}')  # Log the extracted text
    prices = re.findall(r"\d+\.\d+", text)
    
    if len(prices) >= 4:
        open_price = float(prices[0])
        high_price = float(prices[1])
        low_price = float(prices[2])
        close_price = float(prices[3])
        logging.info(f"Precios extraídos: Open={open_price}, High={high_price}, Low={low_price}, Close={close_price}")
        return open_price, high_price, low_price, close_price
    
    logging.warning('No se pudieron extraer suficientes precios.')
    return None, None, None, None

# Selección de la moneda antes de comenzar
print("Seleccione la moneda para predecir:")
for idx, moneda in enumerate(monedas, 1):
    print(f"{idx}. {moneda}")
selected_index = int(input("Ingrese el número de la moneda: ")) - 1
selected_moneda = monedas[selected_index]

# Cargar el modelo y el scaler guardados
model_file = f'lstm_model_{selected_moneda.replace("/", "_").replace(" ", "_")}.h5'
scaler_file = f'scaler_{selected_moneda.replace("/", "_").replace(" ", "_")}.pkl'

model = load_model(model_file)
scaler = joblib.load(scaler_file)

feedback_model_file = f'feedback_model_{selected_moneda.replace("/", "_").replace(" ", "_")}.h5'

def make_prediction(features):
    moneda_feature = [1 if m == selected_moneda else 0 for m in monedas]
    features_combined = list(features) + moneda_feature
    features_scaled = scaler.transform([features_combined])
    features_scaled = np.reshape(features_scaled, (1, 1, len(features_combined)))
    prediction_scaled = model.predict(features_scaled)
    prediction = scaler.inverse_transform([[0] * (len(features_combined) - 1) + [prediction_scaled[0][0]]])
    return round(prediction[0][-1], 5)

def automate_click(decision):
    try:
        if decision == "Subir":
            pyautogui.click(x=1830, y=481)
        elif decision == "Bajar":
            pyautogui.click(x=1827, y=543)
        time.sleep(0.5)  # Espera 0.5 segundos para asegurarse de que el clic se registre
        pyautogui.moveTo(return_coords)  # Mover el cursor a las coordenadas especificadas
        logging.info(f"Automated click executed: {decision}")
    except Exception as e:
        logging.error(f"Error en la automatización de clic: {e}")

def check_operation_result(initial_close_price, decision, predicted_close_price, tolerance=0.00005):
    time.sleep(61)  # Esperar 1 minuto para capturar los nuevos valores después de la operación
    _, _, _, final_close_price = extract_prices_from_image()
    logging.info(f"Cotización de cierre inicial: {initial_close_price}, Predicción de cierre: {predicted_close_price}, Cotización de cierre final: {final_close_price}")
    
    # Evaluar si la predicción fue exitosa según la similitud con el valor de cierre final
    if abs(predicted_close_price - final_close_price) <= tolerance:
        return True  # Ganamos
    else:
        return False  # Perdimos

def play_sound(result):
    if sound_enabled:
        if result:  # Ganamos
            winsound.Beep(1000, 500)  # Sonido agudo (1 kHz) durante 500 ms
        else:  # Perdimos
            winsound.Beep(500, 500)  # Sonido grave (500 Hz) durante 500 ms

def train_feedback_model(features, result):
    if os.path.exists(feedback_model_file):
        feedback_model = load_model(feedback_model_file)
        feedback_model.compile(optimizer='adam', loss='binary_crossentropy')  # Recompilar el modelo
    else:
        feedback_model = Sequential()
        feedback_model.add(Dense(25, input_dim=len(features), activation='relu'))
        feedback_model.add(Dropout(0.3))  # Regularización para evitar sobreajuste
        feedback_model.add(Dense(1, activation='sigmoid'))
        feedback_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    X = np.array(features).reshape(1, -1)
    y = np.array([1 if result else 0])
    
    feedback_model.fit(X, y, epochs=10, verbose=0)  # Incrementar los epochs para un aprendizaje más profundo
    feedback_model.save(feedback_model_file)
    logging.info(f"Modelo retroalimentado guardado: {'Ganamos' if result else 'Perdimos'}")

def analyze_loss():
    logging.info("Analizando la pérdida...")
    time.sleep(300)  # Pausa de 5 minutos para analizar antes de la siguiente operación
    logging.info("Análisis completado. Reanudando operaciones...")

def should_pause():
    if len(result_history) < 10:
        return False  # No hay suficientes datos para decidir

    win_count = sum(result_history)
    loss_count = len(result_history) - win_count

    win_rate = win_count / len(result_history)
    
    logging.info(f"Análisis de rendimiento reciente: Ganadas: {win_count}, Perdidas: {loss_count}, Tasa de ganancia: {win_rate:.2f}")
    
    if win_rate < 0.5:
        logging.warning("El rendimiento reciente es bajo. Pausando para analizar...")
        return True
    
    return False

def capture_and_predict():
    global saldo
    wins = 0
    losses = 0
    total_operations = 0

    while saldo < 8000 and saldo > 500:
        try:
            if should_pause():
                print("Esperando 5 minutos antes de continuar con la próxima operación...")
                analyze_loss()  # Pausa si el rendimiento reciente es bajo

            print(f"Iniciando captura de datos para {selected_moneda} por 3 minutos...")
            data = []
            start_time = time.time()
            while time.time() - start_time < 180:  # Capturar datos durante 3 minutos
                remaining_time = 180 - (time.time() - start_time)
                print(f"Tiempo restante para capturar datos: {remaining_time:.2f} segundos")
                
                open_price, high_price, low_price, close_price = extract_prices_from_image()
                if open_price is not None:
                    data.append([open_price, high_price, low_price, close_price])
                time.sleep(1)  # Captura cada segundo

            if data:
                avg_data = np.mean(data, axis=0)  # Promediar los valores capturados
                print(f'Promedio de datos capturados: Open={avg_data[0]}, High={avg_data[1]}, Low={avg_data[2]}, Close={avg_data[3]}')
                
                prediction = make_prediction(avg_data)
                print(f'Predicción del precio de cierre para {selected_moneda}: {prediction}')
                
                if prediction > avg_data[3]:
                    decision = "Subir"
                    print("Decisión: Subir porque la predicción del precio es mayor que el precio de cierre promedio.")
                else:
                    decision = "Bajar"
                    print("Decisión: Bajar porque la predicción del precio es menor o igual que el precio de cierre promedio.")
                
                automate_click(decision)
                
                # Verificar si ganamos o perdimos la operación
                result = check_operation_result(avg_data[3], decision, prediction)
                result_history.append(1 if result else 0)
                
                if result:
                    saldo += ganancia_por_victoria
                    wins += 1
                    logging.info(f"Resultado: Ganamos. Saldo actual: {saldo}")
                else:
                    saldo -= monto_apuesta
                    losses += 1
                    logging.info(f"Resultado: Perdimos. Saldo actual: {saldo}")
                    analyze_loss()  # Pausa y análisis después de una pérdida
                
                play_sound(result)  # Reproducir sonido según el resultado
                
                total_operations += 1
                win_percentage = (wins / total_operations) * 100
                logging.info(f"Operaciones Ganadas: {wins}, Operaciones Perdidas: {losses}")
                logging.info(f"Porcentaje de operaciones ganadas: {win_percentage:.2f}%")
                
                train_feedback_model(avg_data, result)
                
            else:
                logging.warning("No se pudieron capturar suficientes datos.")
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")

# Iniciar la captura y predicción
print("Iniciando captura y predicción...")
capture_and_predict()
