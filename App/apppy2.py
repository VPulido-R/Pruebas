import tkinter as tk
from datetime import datetime
import threading
import time
import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import pickle

# -------------------- CARGAR ENCODES --------------------
print("[INFO] Cargando encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# -------------------- CONFIG CÁMARA --------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (340, 580)}))
picam2.start()

cv_scaler = 5  # escala para procesar menos pixels y aumentar velocidad

# -------------------- VARIABLES --------------------
latest_name = "Esperando detección..."
latest_time = ""   # <- nueva variable
face_locations = []
face_encodings = []
face_names = []

# -------------------- FUNCIONES --------------------
#----------------------DETECCION DE ROSTROS------------------
def process_frame(frame):
    global face_locations, face_encodings, face_names
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return frame
#-------------------MUESTRA NOMBRE, FECHA Y HORA----------------------
def detection_loop():
    global latest_name, latest_time
    clear_delay = 0.5  # segundos para borrar el nombre si no hay detección
    last_detect_time = None
    
    while True:
        frame = picam2.capture_array()
        _ = process_frame(frame)
        if face_names:
            latest_name = ", ".join(face_names)
            latest_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # <- guarda fecha y hora
            last_detect_time = datetime.now()
        else:
            if last_detect_time and (datetime.now() - last_detect_time).total_seconds() > clear_delay:
                latest_name = "Esperando detección..."
                latest_time = ""  # <- limpia hora
                last_detect_time = None
        time.sleep(0.01)

def update_label():
    label.config(text=latest_name)
    time_label.config(text=latest_time)  # <- actualiza fecha/hora
    root.after(200, update_label)

# -------------------- TKINTER UI --------------------
root = tk.Tk()
root.title("Reconocimiento Facial")
root.geometry("800x400")
root.resizable(False, False)

label = tk.Label(root, text=latest_name, font=("Arial", 48))
label.pack(expand=True)

time_label = tk.Label(root, text=latest_time, font=("Arial", 28))  # <- nuevo label debajo
time_label.pack()

# -------------------- HILOS --------------------
threading.Thread(target=detection_loop, daemon=True).start()
root.after(200, update_label)

# -------------------- CIERRE --------------------
def on_close():
    picam2.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
