from flask import Flask, Response, render_template_string
import sqlite3
from datetime import datetime
import cv2
import atexit
import recognizer

DB = "database.db"

app = Flask(__name__)

# Crea tabla si no existe
def ensure_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()
ensure_db()

def log_detection(name):
    if not name:
        return
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO detections(name, timestamp) VALUES(?,?)",
              (name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

@app.route("/")
def index():
    # HTML mínimo inline para no depender de templates
    html = """
    <h1>Asistencia - Video en vivo</h1>
    <img src="/video" width="640" height="480" />
    <p>Registros: <a href="/registros">/registros</a></p>
    """
    return render_template_string(html)

def gen_frames():
    while True:
        frame, name = recognizer.process_frame()
        if name:  # se guarda UNA vez por persona cada 60s
            log_detection(name)
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/registros")
def registros():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM detections ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    items = "".join(f"<li>{ts} - {name}</li>" for name, ts in rows)
    return f"<h1>Registros</h1><ul>{items}</ul>"

# cerrar cámara al salir
atexit.register(recognizer.shutdown)

if __name__ == "__main__":
    # IMPORTANTE: desactivar reloader para que NO abra la cámara dos veces
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
