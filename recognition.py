import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from keras_facenet import FaceNet

from database import load_faces   # ONLY loading, no writes

# ---------------- TF SAFETY ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.keras.backend.clear_session()

# ---------------- LOAD MODELS SAFELY ----------------
detector = MTCNN()

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = FaceNet()
    return _embedder

embedder = get_embedder()

# ---------------- LOAD KNOWN FACES (SQLITE) ----------------
known_embeddings, known_names = load_faces()

if len(known_embeddings) > 0:
    known_embeddings = known_embeddings.astype("float32")
    known_embeddings /= np.linalg.norm(
        known_embeddings, axis=1, keepdims=True
    )

# ---------------- FACE DECISION THRESHOLD ----------------
# Lower = stricter (0.7–0.8 is realistic for FaceNet)
THRESHOLD = 0.75


# ---------------- MAIN RECOGNITION FUNCTION ----------------
def recognize_face(image_bgr):
    """
    Input:
        image_bgr (np.ndarray): OpenCV BGR image

    Output:
        annotated_image (np.ndarray)
        result (dict): { name, confidence }
    """

    image = image_bgr.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(rgb)

    # ❌ No face detected
    if not faces:
        return image, {
            "name": "No face detected",
            "confidence": 0
        }

    # ✅ Single-person flow → pick largest face
    face = max(faces, key=lambda f: f["box"][2] * f["box"][3])
    x, y, w, h = face["box"]
    x, y = max(0, x), max(0, y)

    face_img = rgb[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")

    # ---------------- EMBEDDING ----------------
    embedding = embedder.embeddings([face_img])[0]
    embedding /= np.linalg.norm(embedding)

    # ---------------- MATCHING ----------------
    name = "Unknown"
    confidence = 0
    color = (0, 0, 255)  # red

    if len(known_embeddings) > 0:
        distances = np.linalg.norm(known_embeddings - embedding, axis=1)
        min_dist = float(np.min(distances))
        idx = int(np.argmin(distances))

        # ✅ DISTANCE decides identity (LOGIC)
        if min_dist < THRESHOLD:
            name = known_names[idx]

            # ✅ EXPONENTIAL confidence (UI ONLY)
            confidence = int(np.exp(-min_dist) * 100)
            confidence = max(0, min(confidence, 100))
            color = (0, 255, 0)  # green

    # ---------------- DRAW ----------------
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    label = f"{name} ({confidence}%)"
    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    return image, {
        "name": name,
        "confidence": confidence
    }
