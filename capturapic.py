import cv2
import os
from datetime import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Nombre de la persona
PERSON_NAME = "vanessa"

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)
    
    # Inicializar la cámara
    camera = PiCamera()
    camera.resolution = (640, 480)
    raw_capture = PiRGBArray(camera, size=(640, 480))
    time.sleep(2)  # dejar que la cámara se estabilice

    photo_count = 0
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    # Captura continua de frames
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Capture", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE para capturar
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, image)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):  # Q para salir
            break

        raw_capture.truncate(0)  # limpiar el buffer para el siguiente frame

    camera.close()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)
