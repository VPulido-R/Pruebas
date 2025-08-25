import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

# Borra todos los registros
c.execute("DELETE FROM detections;")
conn.commit()

# Reinicia el contador de AUTOINCREMENT
c.execute("DELETE FROM sqlite_sequence WHERE name='detections';")
conn.commit()

conn.close()

print("Base de datos vaciada ✅")
