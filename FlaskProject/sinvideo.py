from flask import Flask, jsonify, render_template_string
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import recognizer

DB = "database.db"
COOLDOWN_HOURS = 24
CLEAR_DELAY = 3  # segundos para borrar nombre si no hay detección

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

last_seen = {}
latest_names = []  # nombres detectados recientemente
last_detect_time = None  # timestamp de última detección

def log_detection(name):
    if not name:
        return
    now = datetime.now()
    if name in last_seen and now - last_seen[name] < timedelta(hours=COOLDOWN_HOURS):
        return
    last_seen[name] = now
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO detections(name, timestamp) VALUES(?,?)",
              (name, now.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    print(f"[LOG] {name} registrado a las {now}")

# -------------------- DETECCIÓN EN BACKGROUND --------------------
def detection_loop():
    global latest_names, last_detect_time
    while True:
        frame, name = recognizer.process_frame()
        now = datetime.now()
        if name:
            log_detection(name)
            latest_names = [name]
            last_detect_time = now
        else:
            # Si pasaron más de CLEAR_DELAY segundos desde la última detección, borrar nombre
            if last_detect_time and (now - last_detect_time).total_seconds() > CLEAR_DELAY:
                latest_names = []
        time.sleep(0.1)

threading.Thread(target=detection_loop, daemon=True).start()

# -------------------- PÁGINA WEB --------------------
@app.route("/")
def index():
    html = """
    <html>
    <head>
      <title>Asistencia Facial</title>
      <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { font-size: 60px; color: #333; }
      </style>
    </head>
    <body>
      <h1 id="nombre">Esperando detección...</h1>

      <script>
        function actualizar() {
          fetch('/nombre')
            .then(res => res.json())
            .then(data => {
              const h1 = document.getElementById('nombre');
              h1.textContent = data.nombre || 'Esperando detección...';
            });
        }
        setInterval(actualizar, 1000);
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/nombre")
def nombre():
    return jsonify({"nombre": latest_names[0] if latest_names else ""})

# -------------------- CIERRE --------------------
import atexit
atexit.register(recognizer.shutdown)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
