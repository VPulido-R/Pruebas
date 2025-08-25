from flask import Flask, render_template, Response
import sqlite3
from datetime import datetime
import recognizer
import cv2

app = Flask(__name__)

def log_detection(name):
    if name is None:
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
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registros')
def registros():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM detections ORDER BY timestamp DESC")
    data = c.fetchall()
    conn.close()
    html = "<h1>Registros de Asistencia</h1><ul>"
    for name, ts in data:
        html += f"<li>{ts} - {name}</li>"
    html += "</ul>"
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
