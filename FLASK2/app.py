import tkinter as tk
from tkinter import ttk
from datetime import datetime
import threading
import time
import requests
import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import pickle

# -------------------- CONFIG --------------------
SERVER_URL = "http://TU_IP_LOCAL:5000/registro"  # reemplaza TU_IP_LOCAL con la IP de tu PC/servidor local
cv_scaler = 5  # reducir resolución para acelerar
clear_delay = 0.5  # tiempo para borrar nombre si no hay detección

# -------------------- CARGAR ENCODES --------------------
print("[INFO] Cargando encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# -------------------- INICIALIZAR CÁMARA --------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (340, 580)}))
picam2.start()

# -------------------- VARIABLES --------------------
latest_name = "Esperando detección..."
latest_time = ""
face_names = []

# -------------------- FUNCIONES --------------------
def process_frame(frame):
    global face_names
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')
    
    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Desconocido"
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_idx = np.argmin(face_distances)
        if matches[best_idx]:
            name = known_face_names[best_idx]
        face_names.append(name)
    return frame

def send_log(name, timestamp):
    try:
        requests.post(SERVER_URL, json={"name": name, "timestamp": timestamp}, timeout=2)
    except Exception as e:
        print("Error enviando log:", e)

def detection_loop():
    global latest_name, latest_time
    last_detect_time = None
    while True:
        frame = picam2.capture_array()
        _ = process_frame(frame)
        if face_names:
            detected_name = ", ".join(face_names)
            detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            latest_name = detected_name
            latest_time = detected_time
            last_detect_time = datetime.now()
            
            # actualizar tabla local
            root.after(0, lambda n=detected_name, t=detected_time: add_to_table(n, t))
            
            # enviar al servidor local
            threading.Thread(target=lambda: send_log(detected_name, detected_time), daemon=True).start()
        else:
            if last_detect_time and (datetime.now() - last_detect_time).total_seconds() > clear_delay:
                latest_name = "Esperando detección..."
                latest_time = ""
                last_detect_time = None
        time.sleep(0.01)

def update_label():
    label.config(text=latest_name)
    time_label.config(text=latest_time)
    root.after(200, update_label)

def add_to_table(name, timestamp):
    table.insert("", "end", values=(name, timestamp))

# -------------------- TKINTER UI --------------------
root = tk.Tk()
root.title("Reconocimiento Facial")
root.geometry("800x600")
root.resizable(False, False)

label = tk.Label(root, text=latest_name, font=("Arial", 32))
label.pack(pady=10)

time_label = tk.Label(root, text=latest_time, font=("Arial", 20))
time_label.pack(pady=5)

frame_table = tk.Frame(root)
frame_table.pack(expand=True, fill="both", pady=10)

table = ttk.Treeview(frame_table, columns=("Nombre", "Fecha/Hora"), show="headings", height=8)
table.heading("Nombre", text="Nombre")
table.heading("Fecha/Hora", text="Fecha y Hora")
table.column("Nombre", anchor="center", width=200)
table.column("Fecha/Hora", anchor="center", width=300)
table.pack(expand=True, fill="both")

# -------------------- HILOS --------------------
threading.Thread(target=detection_loop, daemon=True).start()
root.after(200, update_label)

# -------------------- CIERRE --------------------
def on_close():
    picam2.stop()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
