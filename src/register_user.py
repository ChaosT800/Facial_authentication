import cv2
import os

def register_user(name, user_class):
    path = f"dataset/{user_class}/{name}"
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("Press SPACE to capture images, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register User", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            img_path = f"{path}/{count}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Saved {img_path}")
            count += 1

            if count >= 20:
                break

    cap.release()
    cv2.destroyAllWindows()