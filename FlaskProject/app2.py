from flask import Flask, Response, render_template_string
import sqlite3
from datetime import datetime, timedelta
import cv2
import atexit
import threading
import recognizer

DB = "database.db"
COOLDOWN_HOURS = 24  # una vez cada 24h por persona

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

# Diccionario para manejar cooldown
last_seen = {}

def log_detection(name):
    if not name:
        return
    now = datetime.now()
    # cooldown: si ya se registró en las últimas 24h, no guardar
    if name in last_seen and now - last_seen[name] < timedelta(hours=COOLDOWN_HOURS):
        return
    last_seen[name] = now
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO detections(name, timestamp) VALUES(?,?)",
        (name, now.strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()
    print(f"[LOG] {name} registrado a las {now}")

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
latest_frame = None
latest_name = None

def capture_loop():
    global latest_frame, latest_name
    while True:
        frame, name = recognizer.process_frame()
        latest_frame = frame
        latest_name = name

threading.Thread(target=capture_loop, daemon=True).start()

def gen_frames():
    global latest_frame, latest_name
    while True:
        if latest_frame is None:
            continue
        if latest_name:
            log_detection(latest_name)
            latest_name = None
        ok, buf = cv2.imencode(".jpg", latest_frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# -------------------- CIERRE DE CÁMARA --------------------
atexit.register(recognizer.shutdown)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
