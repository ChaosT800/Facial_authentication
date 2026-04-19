import cv2
import os
os.environ['TORCH_HOME'] = 'models'
import pickle
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from collections import deque, Counter
import datetime

# -------- INIT --------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

with open(r"E:\AICTE\Varun\face_auth_system\models\embeddings.pkl", "rb") as f:
    database = pickle.load(f)

THRESHOLD = float(input("Enter threshold (0.6–0.8 recommended): "))
FRAME_BUFFER = 5

recent_predictions = deque(maxlen=FRAME_BUFFER)

# -------- FUNCTIONS --------
def cosine_distance(a, b):
    return 1 - np.dot(a, b)

def get_stable_prediction(predictions):
    if len(predictions) == 0:
        return None
    count = Counter(predictions)
    return count.most_common(1)[0][0]

def log_access(name, status):
    with open("access_log.txt", "a") as f:
        time = datetime.datetime.now()
        f.write(f"{time} - {name} - {status}\n")

def recognize(face_embedding):
    min_dist = float("inf")
    identity = None
    identity_class = None

    for entry in database:
        dist = cosine_distance(face_embedding, entry["embedding"])
        if dist < min_dist:
            min_dist = dist
            identity = entry["name"]
            identity_class = entry["class"]

    if min_dist < THRESHOLD:
        confidence = (1 - min_dist) * 100
        return identity, identity_class, confidence
    else:
        return "Unknown", "Unknown", 0

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

INPUT_CLASS = input("Enter required class: ")

print("Starting camera...")

# -------- LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = face / 255.0

            face_tensor = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0).to(device)

            embedding = model(face_tensor).detach().cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)

            name, person_class, confidence = recognize(embedding)

            recent_predictions.append(name)
            stable_name = get_stable_prediction(recent_predictions)

            if stable_name == "Unknown":
                label = "ACCESS DENIED"
                color = (0, 0, 255)
                log_access("Unknown", "DENIED")

            else:
                if person_class == INPUT_CLASS:
                    label = f"{stable_name} | GRANTED ({confidence:.1f}%)"
                    color = (0, 255, 0)
                    log_access(stable_name, "GRANTED")
                else:
                    label = f"{stable_name} | DENIED"
                    color = (0, 0, 255)
                    log_access(stable_name, "DENIED")

            # UI box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1-40), (x2, y1), color, -1)

            cv2.putText(frame, label, (x1+5, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Confidence bar
            bar_length = int((confidence / 100) * (x2 - x1))
            cv2.rectangle(frame, (x1, y2+5), (x1 + bar_length, y2+15), color, -1)

    cv2.imshow("Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()