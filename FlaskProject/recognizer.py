import face_recognition
import numpy as np
from picamera2 import Picamera2
import pickle
import time
import cv2

# Cargar encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Cámara
picam2 = None
cv_scaler = 1
last_seen = {}  # para evitar registros duplicados

def init_camera():
    global picam2
    if picam2 is None:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (480, 480)}
        ))
        picam2.start()
    return picam2

def process_frame():
    """
    Devuelve:
      - frame con rectángulos y nombres
      - nombre detectado (None si no hay detección o ya registrado recientemente)
    """
    global last_seen
    cam = init_camera()
    frame = cam.capture_array()

    rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    name_detected = None
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            now = time.time()
            if name not in last_seen or now - last_seen[name] > 60:
                last_seen[name] = now
                name_detected = name

        # Dibujar rectángulo y nombre
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left-3, top-35), (right+3, top), (244, 42, 3), cv2.FILLED)
        cv2.putText(frame, name, (left+6, top-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)

    return frame, name_detected
