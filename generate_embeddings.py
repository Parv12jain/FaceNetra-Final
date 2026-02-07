import os
import cv2
from  keras_facenet import FaceNet
import pickle
import numpy as np

# model
embedder  = FaceNet()
DATASET_DIR = "processed_faces"   # cropped faces folder
EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

embeddings = []
labels = []

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))

        img = img.astype("float32")
        img = np.expand_dims(img, axis=0)

        # Generate embedding
        embedding = embedder.embeddings(img)[0]

        embeddings.append(embedding)
        labels.append(person_name)

# Convert to arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

with open(os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl"), "wb") as f:
    pickle.dump((embeddings, labels), f)

print("FaceNet embeddings generated and saved successfully")
print(f"Total embeddings: {len(embeddings)}")