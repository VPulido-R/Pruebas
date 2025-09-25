from flask import Flask, request, jsonify

app = Flask(__name__)

# Guardaremos los registros en memoria
registros = []

# Endpoint para recibir los logs
@app.route("/registro", methods=["POST"])
def registro():
    data = request.get_json()
    if not data or "name" not in data or "timestamp" not in data:
        return jsonify({"error": "Bad request"}), 400

    registros.append(data)
    return jsonify({"status": "ok", "data": data}), 200

# Endpoint para ver los logs en JSON
@app.route("/ver", methods=["GET"])
def ver():
    return jsonify(registros)

# Endpoint simple para verlos en HTML
@app.route("/")
def home():
    html = "<h1>Registros de caras</h1><ul>"
    for r in registros:
        html += f"<li>{r['name']} - {r['timestamp']}</li>"
    html += "</ul>"
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
