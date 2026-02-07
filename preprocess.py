import os
import cv2
import numpy as np
from mtcnn import MTCNN

# intizaling  face detector
detector = MTCNN()

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_faces"
IMG_SIZE = (160, 160)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)

    face = img_rgb[y:y+h, x:x+w]
    face = cv2.resize(face, IMG_SIZE)
    face = face / 255.0  # normalize

    return face

for person in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person)
    save_path = os.path.join(OUTPUT_DIR, person)
    os.makedirs(save_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        face = extract_face(img_path)

        if face is not None:
            save_img = (face * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(save_path, img_name),
                cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            )

print("Face preprocessing completed!")
