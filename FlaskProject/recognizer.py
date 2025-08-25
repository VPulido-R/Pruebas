import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import pickle
import time

print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Inicializa cámara
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": 'XRGB8888', "size": (480, 480)}
))
picam2.start()

cv_scaler = 1
face_locations = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame():
    global face_locations, face_names, fps, frame_count, start_time
    
    frame = picam2.capture_array()

    # Resize y conversión
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Detección
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    face_names = []
    name_detected = None
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            name_detected = name
        face_names.append(name)

    # Dibujar resultados
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left-3, top-35), (right+3, top), (244, 42, 3), cv2.FILLED)
        cv2.putText(frame, name, (left+6, top-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 1)

    # FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame, name_detected
