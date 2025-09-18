from flask import Flask, jsonify, render_template_string
import threading
import time
import recognizer
from datetime import datetime

app = Flask(__name__)
CLEAR_DELAY = 3  # segundos para borrar nombre si no hay detección

latest_name = ""
last_detect_time = None

# -------------------- DETECCIÓN EN BACKGROUND --------------------
def detection_loop():
    global latest_name, last_detect_time
    while True:
        frame, name = recognizer.process_frame()
        now = datetime.now()
        if name:
            latest_name = name
            last_detect_time = now
        else:
            if last_detect_time and (now - last_detect_time).total_seconds() > CLEAR_DELAY:
                latest_name = ""
        time.sleep(0.1)

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
