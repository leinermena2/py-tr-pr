import cv2
import numpy as np
from PIL import ImageGrab

# Coordenadas del área donde se encuentra el gráfico de velas
candlestick_area_coords = (87, 164, 1605, 937)  # Coordenadas para recortar la región de las velas

# Coordenadas ajustadas para capturar la región de precios según la imagen proporcionada
price_area_coords = (139, 864, 239, 955)  # Ajustadas según lo proporcionado

def improve_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return sharpened

def capture_full_screen_and_crop(crop_coords, full_image):
    x1, y1, x2, y2 = crop_coords
    cropped_image = full_image[y1:y2, x1:x2]  # Recorta la región de interés
    return cropped_image

def preprocess_image_for_candles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return processed

def detect_candles(image):
    processed_image = preprocess_image_for_candles(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 20 and w < 20:  # Ajustar estos valores según el tamaño esperado de las velas
            candles.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Detected Candles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return candles

def capture_and_show_images():
    # Captura de pantalla completa
    full_screen = np.array(ImageGrab.grab())
    
    # Recortar la región de las velas desde la pantalla completa
    screen_candles = capture_full_screen_and_crop(candlestick_area_coords, full_screen)
    detect_candles(screen_candles)
    
    # Recortar la región de precios desde la pantalla completa
    screen_prices = capture_full_screen_and_crop(price_area_coords, full_screen)
    improved_prices = improve_image_quality(screen_prices)
    
    cv2.imshow('Price Area (Improved)', improved_prices)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecutar la función para capturar y mostrar las imágenes
capture_and_show_images()
