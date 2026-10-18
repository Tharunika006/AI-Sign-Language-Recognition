import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# Sign name
sign = input("Enter sign name (example: HELLO): ").strip().upper()

# CSV file
os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{sign}.csv"

# Create CSV file
file = open(file_path, "w", newline="")
writer = csv.writer(file)

# CSV header
header = ["label"]

for i in range(1, 22):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

writer.writerow(header)

cap = cv2.VideoCapture(0)

timestamp = 0
count = 0

print("\nGet ready...")
print("Show your hand clearly.")
print("Press Q to stop collecting.\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not detected")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        row = [sign]

        for landmark in hand:
            row.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        writer.writerow(row)
        count += 1

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.putText(
        frame,
        f"Sign: {sign}  Samples: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

file.close()
cap.release()
landmarker.close()
cv2.destroyAllWindows()

print(f"\nData collection completed!")
print(f"Samples collected: {count}")
print(f"Saved to: {file_path}")