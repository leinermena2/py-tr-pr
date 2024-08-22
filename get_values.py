import cv2
import numpy as np
from PIL import ImageGrab

# Coordenadas del área donde se encuentran los valores open, high, low, close
price_area_coords = (140, 877, 233, 956)  # (x1, y1, x2, y2)

# Capturar la región de precios
screen_price = np.array(ImageGrab.grab(bbox=price_area_coords))

# Guardar la imagen capturada en el sistema de archivos para visualizarla
cv2.imwrite('price_area_capture.png', screen_price)

# Mostrar la imagen capturada
cv2.imshow("Captured Price Area", screen_price)
cv2.waitKey(0)
cv2.destroyAllWindows()
