from flask import Flask, render_template, Response
import cv2
import sqlite3
from datetime import datetime
import recognizer  # tu módulo de reconocimiento facial

app = Flask(__name__)

# Conectar a la base de datos
def log_detection(name):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO detections (name, timestamp) VALUES (?, ?)",
              (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Cámara
camera = cv2.VideoCapture(0)  # 0 = cámara USB / PiCam

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Aquí usas tu código de reconocimiento facial
            name, frame = recognizer.detect_face(frame)

            # Si se detecta una persona conocida, guardamos en DB
            if name is not None:
                log_detection(name)

            # Convertimos a formato web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
