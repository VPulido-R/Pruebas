from flask import Flask, Response, render_template_string
import sqlite3
from datetime import datetime
import cv2
import atexit
import recognizer

DB = "database.db"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80  # 0-100, ajustar si quieres más rápido

app = Flask(__name__)

# -------------------- BASE DE DATOS --------------------
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

# -------------------- PÁGINAS WEB --------------------
@app.route("/")
def index():
    html = """
    <h1>Asistencia - Video en vivo</h1>
    <img src="/video" width="640" height="480" />
    <p>Registros: <a href="/registros">/registros</a></p>
    """
    return render_template_string(html)

@app.route("/registros")
def registros():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT name, timestamp FROM detections ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    items = "".join(f"<li>{ts} - {name}</li>" for name, ts in rows)
    return f"<h1>Registros</h1><ul>{items}</ul>"

# -------------------- STREAM DE VIDEO --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Procesamiento de reconocimiento facial
        frame, name = recognizer.process_frame(frame)
        if name:
            log_detection(name)

        # Codificar frame como JPEG con calidad ajustable
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------- CIERRE --------------------
atexit.register(lambda: (cap.release(), recognizer.shutdown()))

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
