from flask import Flask, jsonify, render_template_string
import threading
import time
import recognizer
from datetime import datetime

app = Flask(__name__)
CLEAR_DELAY = 3       # segundos para borrar nombre si no hay detección
PROCESS_INTERVAL = 0.5  # segundos entre detecciones

latest_name = ""
last_detect_time = None
last_process_time = 0

# -------------------- DETECCIÓN EN BACKGROUND --------------------
def detection_loop():
    global latest_name, last_detect_time, last_process_time
    while True:
        now = time.time()
        if now - last_process_time >= PROCESS_INTERVAL:
            last_process_time = now

            # Tomar frame y reducir resolución para acelerar detección
            frame, name = recognizer.process_frame()
            
            if name:
                latest_name = name
                last_detect_time = datetime.now()
            else:
                if last_detect_time and (datetime.now() - last_detect_time).total_seconds() > CLEAR_DELAY:
                    latest_name = ""
                    last_detect_time = None
        time.sleep(0.05)  # pequeño sleep para no saturar CPU

threading.Thread(target=detection_loop, daemon=True).start()

# -------------------- PÁGINA WEB --------------------
@app.route("/")
def index():
    html = """
    <html>
    <head>
      <title>Reconocimiento Facial</title>
      <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { font-size: 80px; color: #333; }
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
        setInterval(actualizar, 500); // refresco cada 0.5s
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route("/nombre")
def nombre():
    return jsonify({"nombre": latest_name})

# -------------------- CIERRE --------------------
import atexit
atexit.register(recognizer.shutdown)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
