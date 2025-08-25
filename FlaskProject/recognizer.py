import pickle
import time
import cv2
import numpy as np
import face_recognition
from picamera2 import Picamera2

# Carga encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
KNOWN_ENCODINGS = data["encodings"]
KNOWN_NAMES = data["names"]

# Estado cámara y anti-duplicados
_picam2 = None
_last_seen = {}   # name -> epoch seconds
DEDUP_SECONDS = 60

def init_camera():
    """Inicializa la cámara una sola vez en formato 3 canales."""
    global _picam2
    if _picam2 is None:
        cam = Picamera2()
        cam.configure(
            cam.create_preview_configuration(
                main={"format": "BGR888", "size": (640, 480)}
            )
        )
        cam.start()
        _picam2 = cam
    return _picam2

def shutdown():
    """Para cerrar la cámara al terminar."""
    global _picam2
    if _picam2 is not None:
        try:
            _picam2.stop()
        except Exception:
            pass
        _picam2 = None

def process_frame():
    """
    Captura un frame, dibuja boxes/nombres y devuelve:
      frame_bgr (np.ndarray), detected_name (str|None si ya se registró hace poco o es Unknown)
    """
    cam = init_camera()
    frame = cam.capture_array()  # BGR (640x480x3)

    # Reconocimiento
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encs  = face_recognition.face_encodings(rgb, boxes)
    detected_name = None

    for (top, right, bottom, left), enc in zip(boxes, encs):
        matches = face_recognition.compare_faces(KNOWN_ENCODINGS, enc)
        name = "Unknown"
        if len(matches):
            dists = face_recognition.face_distance(KNOWN_ENCODINGS, enc)
            i = int(np.argmin(dists))
            if matches[i]:
                name = KNOWN_NAMES[i]

        # anti-duplicados
        if name != "Unknown":
            now = time.time()
            if name not in _last_seen or now - _last_seen[name] > DEDUP_SECONDS:
                _last_seen[name] = now
                detected_name = name  # se registra UNA vez

        # dibujar
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
        cv2.rectangle(frame, (left, top - 28), (right, top), (0, 200, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 5, top - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, detected_name
