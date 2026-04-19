import os
os.environ['TORCH_HOME'] = 'models'
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

# ---------------- CONFIG ----------------
DATASET_PATH = r"E:\AICTE\Varun\face_auth_system\dataset"
OUTPUT_FILE = r"E:\AICTE\Varun\face_auth_system\models\embeddings.pkl"
USE_AVERAGE_EMBEDDING = True  # 🔥 Set True for better stability

# ---------------- INIT ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Using device: {device}")

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings_db = []

# ---------------- FUNCTIONS ----------------
def process_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[WARNING] Skipping invalid image: {image_path}")
        return None

    face = mtcnn(img)

    if face is None:
        print(f"[WARNING] No face detected: {image_path}")
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face).cpu().numpy()[0]

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

# ---------------- MAIN ----------------
print("[INFO] Processing dataset...")

if not os.path.exists(DATASET_PATH):
    print("[ERROR] Dataset folder not found!")
    exit()

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    # ✅ Skip non-directories
    if not os.path.isdir(class_path):
        print(f"[SKIP] Not a folder: {class_path}")
        continue

    print(f"\n[CLASS] {class_name}")

    for person in os.listdir(class_path):
        person_path = os.path.join(class_path, person)

        # ✅ Skip non-directories
        if not os.path.isdir(person_path):
            print(f"[SKIP] Not a person folder: {person_path}")
            continue

        print(f"  [PERSON] {person}")

        person_embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # ✅ Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"    [SKIP] Not an image: {img_name}")
                continue

            emb = process_image(img_path)

            if emb is not None:
                person_embeddings.append(emb)

        # ❌ No valid images
        if len(person_embeddings) == 0:
            print(f"  [WARNING] No valid embeddings for {person}")
            continue

        # 🔥 OPTION 1: Average embedding (recommended)
        if USE_AVERAGE_EMBEDDING:
            avg_embedding = np.mean(person_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

            embeddings_db.append({
                "embedding": avg_embedding,
                "name": person,
                "class": class_name
            })

            print(f"  [OK] Stored AVG embedding ({len(person_embeddings)} images)")

        # 🔥 OPTION 2: Store all embeddings
        else:
            for emb in person_embeddings:
                embeddings_db.append({
                    "embedding": emb,
                    "name": person,
                    "class": class_name
                })

            print(f"  [OK] Stored {len(person_embeddings)} embeddings")

# ---------------- SAVE ----------------
if len(embeddings_db) == 0:
    print("[ERROR] No embeddings generated. Check dataset!")
    exit()

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings_db, f)

print(f"\n✅ SUCCESS: {len(embeddings_db)} embeddings saved to {OUTPUT_FILE}")