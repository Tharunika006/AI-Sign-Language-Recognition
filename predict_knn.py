import cv2
import mediapipe as mp
import pickle
import math
import pyttsx3
import threading

from collections import deque, Counter


# =========================================================
# LOAD KNN MODEL
# =========================================================

with open("sign_language_knn.pkl", "rb") as file:
    model = pickle.load(file)

X = model["features"]
y = model["labels"]

print("KNN model loaded!")
print("Training samples:", len(X))
print("Signs:", sorted(set(y)))


# =========================================================
# TEXT TO SPEECH
# =========================================================

def speak(text):

    def voice():

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)

            engine.say(text)
            engine.runAndWait()

            engine.stop()

        except Exception as error:
            print("Voice error:", error)

    threading.Thread(
        target=voice,
        daemon=True
    ).start()


# =========================================================
# NORMALIZE HAND LANDMARKS
# =========================================================

def normalize_landmarks(values):

    landmarks = []

    for i in range(21):

        x = values[i * 3]
        y_value = values[i * 3 + 1]
        z = values[i * 3 + 2]

        landmarks.append([
            x,
            y_value,
            z
        ])

    # Use wrist as the origin
    wrist = landmarks[0]

    for point in landmarks:

        point[0] -= wrist[0]
        point[1] -= wrist[1]
        point[2] -= wrist[2]

    # Calculate hand width
    min_x = min(point[0] for point in landmarks)
    max_x = max(point[0] for point in landmarks)

    width = max_x - min_x

    if width < 0.001:
        width = 1

    normalized = []

    for point in landmarks:

        normalized.extend([
            point[0] / width,
            point[1] / width,
            point[2] / width
        ])

    return normalized


# =========================================================
# EUCLIDEAN DISTANCE
# =========================================================

def distance(a, b):

    total = 0

    for i in range(len(a)):

        difference = a[i] - b[i]
        total += difference * difference

    return math.sqrt(total)


# =========================================================
# KNN PREDICTION
# =========================================================

def predict(sample, k=5):

    distances = []

    for i in range(len(X)):

        d = distance(sample, X[i])

        distances.append((d, y[i]))

    distances.sort(key=lambda item: item[0])

    nearest = distances[:k]

    votes = {}

    for d, label in nearest:

        votes[label] = votes.get(label, 0) + 1

    best_label = max(
        votes,
        key=votes.get
    )

    confidence = (
        votes[best_label] / len(nearest)
    ) * 100

    return best_label, confidence


# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

prediction_history = deque(maxlen=7)


# =========================================================
# SENTENCE BUILDER AND RELEASE SETTINGS
# =========================================================

sentence = []

last_sign = None
stable_count = 0

REQUIRED_FRAMES = 12

# After adding a sign, wait for the hand to leave the frame.
waiting_for_release = False

# Number of consecutive frames without a detected hand.
no_hand_frames = 0

# Require 5 frames without a hand before accepting another sign.
RELEASE_FRAMES = 5


# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# =========================================================
# CAMERA INITIALIZATION
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Camera not detected.")

    landmarker.close()
    raise SystemExit

timestamp = 0

print("Camera started!")
print("Press C to clear text.")
print("Press Q to quit.")


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Unable to read camera frame.")
            break

        # -------------------------------------------------
        # Convert BGR to RGB
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # -------------------------------------------------
        # Detect hand landmarks
        # -------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 1

        current_prediction = None
        confidence = 0

        # =================================================
        # HAND FOUND
        # =================================================

        if len(result.hand_landmarks) > 0:

            # Reset the no-hand counter
            no_hand_frames = 0

            hand = result.hand_landmarks[0]

            # ---------------------------------------------
            # Extract 21 hand landmarks
            # ---------------------------------------------

            values = []

            for landmark in hand:

                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # ---------------------------------------------
            # Normalize landmarks
            # ---------------------------------------------

            features = normalize_landmarks(values)

            # ---------------------------------------------
            # KNN prediction using K = 5
            # ---------------------------------------------

            label, confidence = predict(
                features,
                k=5
            )

            # ---------------------------------------------
            # Smooth predictions across recent frames
            # ---------------------------------------------

            prediction_history.append(label)

            counter = Counter(prediction_history)

            stable_label, stable_votes = (
                counter.most_common(1)[0]
            )

            stable_confidence = (
                stable_votes / len(prediction_history)
            ) * 100

            # ---------------------------------------------
            # Accept stable predictions
            # ---------------------------------------------

            if stable_confidence >= 70:

                current_prediction = stable_label

            # =================================================
            # SENTENCE LOGIC
            # =================================================

            # Accept a sign only when not waiting for release.
            if (
                current_prediction is not None
                and not waiting_for_release
            ):

                # A different sign has appeared
                if current_prediction != last_sign:

                    stable_count = 1
                    last_sign = current_prediction

                else:

                    stable_count += 1

                # -----------------------------------------
                # Add the sign after sufficient stable frames
                # -----------------------------------------

                if stable_count == REQUIRED_FRAMES:

                    # Prevent consecutive duplicate words
                    if (
                        len(sentence) == 0
                        or sentence[-1] != current_prediction
                    ):

                        sentence.append(current_prediction)

                        print("Added:", current_prediction)

                        # Speak without blocking the camera
                        speak(current_prediction)

                        # Wait for the hand to be removed
                        waiting_for_release = True

                    # Prevent repeated additions while holding
                    stable_count = REQUIRED_FRAMES + 1

            # =================================================
            # DRAW HAND LANDMARKS
            # =================================================

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y_pixel = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y_pixel),
                    5,
                    (0, 255, 0),
                    -1
                )

        # =================================================
        # NO HAND FOUND
        # =================================================

        else:

            prediction_history.clear()

            last_sign = None
            stable_count = 0

            # Count consecutive frames without a hand
            no_hand_frames += 1

            # Unlock the sentence builder after release
            if no_hand_frames >= RELEASE_FRAMES:

                waiting_for_release = False

        # =================================================
        # DISPLAY CURRENT PREDICTION
        # =================================================

        if current_prediction:

            display_text = (
                "Sign: "
                + current_prediction
                + " "
                + str(int(confidence))
                + "%"
            )

        else:

            display_text = "Sign: ---"

        cv2.putText(
            frame,
            display_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # =================================================
        # DISPLAY INSTRUCTIONS
        # =================================================

        cv2.putText(
            frame,
            "C = Clear Text    Q = Quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # =================================================
        # DISPLAY SENTENCE
        # =================================================

        sentence_text = " ".join(sentence)

        # Black background for sentence
        cv2.rectangle(
            frame,
            (10, frame.shape[0] - 80),
            (frame.shape[1] - 10, frame.shape[0] - 10),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "Text: " + sentence_text,
            (20, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # =================================================
        # SHOW WEBCAM WINDOW
        # =================================================

        cv2.imshow(
            "AI Sign Language Recognition",
            frame
        )

        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # Clear sentence
        if key == ord("c"):

            sentence.clear()

            prediction_history.clear()

            last_sign = None
            stable_count = 0

            print("Sentence cleared!")

        # Quit application
        if key == ord("q"):

            break

finally:

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print("Program closed successfully.")