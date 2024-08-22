# Guarda este contenido en un archivo llamado install_dependencies.sh

# Instalar pip si no lo tienes
sudo apt-get update
sudo apt-get install -y python3-pip

# Instalar Selenium
pip install selenium

# Instalar OpenCV
pip install opencv-python

# Instalar Pillow
pip install pillow

# Instalar Tesseract OCR y la librería pytesseract
pip install pytesseract

# Instalar otras dependencias útiles
pip install numpy

# Si necesitas usar un navegador específico, asegúrate de tener el WebDriver correspondiente
# Para Chrome:
# wget https://chromedriver.storage.googleapis.com/93.0.4577.15/chromedriver_linux64.zip
# unzip chromedriver_linux64.zip
# sudo mv chromedriver /usr
