# database.py
import sqlite3
import numpy as np
import pickle

DB_PATH = "facenetra.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_face(name, embedding):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    blob = pickle.dumps(embedding)

    cur.execute(
        "INSERT INTO faces (name, embedding) VALUES (?, ?)",
        (name, blob)
    )

    conn.commit()
    conn.close()


def load_faces():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name, embedding FROM faces")
    rows = cur.fetchall()
    conn.close()

    embeddings = []
    names = []

    for name, blob in rows:
        embeddings.append(pickle.loads(blob))
        names.append(name)

    if len(embeddings) == 0:
        return np.array([]), []

    return np.array(embeddings), names

def get_registered_users():
    import sqlite3
    conn = sqlite3.connect("facenetra.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT name, COUNT(*) as count
        FROM faces
        GROUP BY name
        ORDER BY name
    """)

    data = cur.fetchall()
    conn.close()
    return data

import sqlite3

DB_PATH = "facenetra.db"

def delete_user(name: str):
    """
    Delete all embeddings for a given user
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM faces WHERE name = ?",
        (name,)
    )

    conn.commit()
    conn.close()
