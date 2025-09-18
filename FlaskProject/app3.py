from flask import Flask, Response, render_template_string
import cv2
import atexit
import recognizer

app = Flask(__name__)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Pasamos el frame al recognizer
        frame, name = recognizer.process_frame(frame)

        # Solo para mostrar
        # Si frame viene en RGB, convertir a BGR
        try:
            frame_to_show = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except:
            frame_to_show = frame

        ok, buf = cv2.imencode(".jpg", frame_to_show)
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    html = """
    <h1>Video en vivo</h1>
    <img src="/video" width="640" height="480" />
    """
    return render_template_string(html)

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

atexit.register(lambda: (cap.release(), recognizer.shutdown()))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
