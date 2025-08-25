from picamera2 import Picamera2
import cv2
import time

# Inicializar cámara
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Esperar a que la cámara se estabilice
time.sleep(2)

# Capturar un frame
frame = picam2.capture_array()

# Mostrar el frame usando OpenCV
cv2.imshow("Test Picamera2", frame)
print("Presiona cualquier tecla en la ventana para cerrar...")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Detener la cámara
picam2.stop()
