import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os
import numpy as np

# Setup paths
known_faces_dir = r'D:\WORD !\Attendance'  # Update this if needed
attendance_file = os.path.abspath("attendance.xlsx")
captured_dir = os.path.abspath("Captured_Images")

# Ensure captured images directory exists
if not os.path.exists(captured_dir):
    os.makedirs(captured_dir)

# Load known face encodings and names
known_faces = []
known_names = []

print("📁 Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(path)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])
        except IndexError:
            print(f"❌ No face found in {filename}. Skipping.")

def capture_image():
    cam = cv2.VideoCapture(0)
    frame = None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        cv2.imshow('📷 Press Space to Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cam.release()
    cv2.destroyAllWindows()

    if frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(captured_dir, f"captured_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Image saved: {save_path}")
        return frame, save_path

    return None, None

def recognize_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)

    if len(encodings) == 0:
        print("❌ No face detected in captured image.")
        return None
    elif len(encodings) > 1:
        print("⚠️ Multiple faces detected. Using the first one.")

    captured_encoding = encodings[0]
    face_distances = face_recognition.face_distance(known_faces, captured_encoding)

    if len(face_distances) == 0:
        return None

    best_match_index = np.argmin(face_distances)
    if face_recognition.compare_faces([known_faces[best_match_index]], captured_encoding)[0]:
        confidence = 1 - face_distances[best_match_index]  # Simple confidence estimate
        print(f"🎯 Match found: {known_names[best_match_index]} (Confidence: {confidence:.2f})")
        return known_names[best_match_index]
    
    return None

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:
        df = pd.read_excel(attendance_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Prevent duplicate entries for the same person on the same day
    if not df[(df["Name"] == name) & (df["Date"] == date)].empty:
        print(f"{name} already marked present today.")
        return

    new_row = {"Name": name, "Date": date, "Time": time}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(attendance_file, index=False)
    print(f"Attendance marked for {name}")

def main():
    print("🔍 Starting Face Recognition Attendance System...")
    image, img_path = capture_image()

    if image is None:
        print("No image captured. Exiting.")
        return

    name = recognize_face(image)
    if name:
        mark_attendance(name)
        print(f"Welcome, {name}!")
    else:
        print(" Face not recognized. Attendance not marked.")

if __name__ == "__main__":
    main()
