from flask import Flask, render_template, Response
import cv2
import sqlite3
from datetime import datetime
import recognizer  # nuestro módulo modificado

app = Flask(__name__)

# Función para guardar detecciones en SQLite
def log_detection(name):
    if name is None or name == "Unknown":
        return
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO detections (name, timestamp) VALUES (?, ?)",
              (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def generate_frames():
    while True:
        frame, detected_name = recognizer.process_frame()

        if detected_name:
            log_detection(detected_name)

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
