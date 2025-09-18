from flask import Flask, jsonify, render_template_string
import threading
import time
import cv2
import recognizer
from datetime import datetime

app = Flask(__name__)
CLEAR_DELAY = 3
PROCESS_INTERVAL = 0.5  # segundos entre detecciones

latest_name = ""
last_detect_time = None
last_process_time = 0

# -------------------- CAPTURA DE CÁMARA --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def detection_loop():
    global latest_name, last_detect_time, last_process_time
    while True:
        now = time.time()
        if now - last_process_time >= PROCESS_INTERVAL:
            last_process_time = now

            ret, frame = cap.read()
            if not ret:
                continue

            # Pasamos el frame al recognizer
            name = recognizer.process_frame(frame)[1]  # asumimos que devuelve (frame, name)
            
            if name:
                latest_name = name
                last_detect_time = datetime.now()
            else:
                if last_detect_time and (datetime.now() - last_detect_time).total_seconds() > CLEAR_DELAY:
                    latest_name = ""
                    last_detect_time = None
        time.sleep(0.05)

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
        setInterval(actualizar, 500);
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
def shutdown():
    cap.release()
    recognizer.shutdown()
atexit.register(shutdown)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
