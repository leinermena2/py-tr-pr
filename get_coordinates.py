import pyautogui

def get_mouse_position():
    try:
        while True:
            x, y = pyautogui.position()
            print(f"Posición actual del mouse: (X: {x}, Y: {y})")
            pyautogui.sleep(1)  # Espera 1 segundo entre lecturas de posición
    except KeyboardInterrupt:
        print("Finalizando lectura de coordenadas.")

print("Mueve el mouse sobre el botón y observa las coordenadas en la consola. Presiona Ctrl+C para finalizar.")
get_mouse_position()
