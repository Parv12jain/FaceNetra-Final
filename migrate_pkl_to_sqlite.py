import pickle
import numpy as np
from database import init_db, save_face

# Initialize DB
init_db()

# Load old embeddings
with open("embeddings/face_embeddings.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

for name, emb in zip(known_names, known_embeddings):
    emb = emb.astype("float32")
    save_face(name, emb)

print("✅ Migration complete: old users moved to SQLite")
