import csv
import glob
import os
import pickle
import math
import random

DATASET_FOLDER = "dataset"

X = []
y = []


# -----------------------------------
# Normalize hand landmarks
# -----------------------------------
def normalize_landmarks(values):

    landmarks = []

    for i in range(21):
        x = values[i * 3]
        y_value = values[i * 3 + 1]
        z = values[i * 3 + 2]

        landmarks.append([x, y_value, z])

    # Wrist as origin
    wrist = landmarks[0]

    for point in landmarks:
        point[0] -= wrist[0]
        point[1] -= wrist[1]
        point[2] -= wrist[2]

    # Find hand width
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


# -----------------------------------
# Create mirrored version
# -----------------------------------
def mirror_landmarks(features):

    mirrored = []

    for i in range(21):

        x = features[i * 3]
        y_value = features[i * 3 + 1]
        z = features[i * 3 + 2]

        # Mirror X axis
        x = -x

        mirrored.extend([
            x,
            y_value,
            z
        ])

    return mirrored


# -----------------------------------
# Load dataset
# -----------------------------------
for file_path in glob.glob(
    os.path.join(DATASET_FOLDER, "*.csv")
):

    with open(file_path, "r", newline="") as file:

        reader = csv.reader(file)

        # Skip header
        next(reader)

        for row in reader:

            if len(row) != 64:
                continue

            label = row[0]

            values = [
                float(value)
                for value in row[1:]
            ]

            # Original sample
            features = normalize_landmarks(values)

            X.append(features)
            y.append(label)

            # Mirrored sample
            mirrored = mirror_landmarks(features)

            X.append(mirrored)
            y.append(label)


# -----------------------------------
# Dataset information
# -----------------------------------
print("Total samples after mirroring:", len(X))
print("Total features:", len(X[0]))
print("Signs:", sorted(set(y)))


# -----------------------------------
# Shuffle
# -----------------------------------
combined = list(zip(X, y))

random.seed(42)
random.shuffle(combined)

X, y = zip(*combined)

X = list(X)
y = list(y)


# -----------------------------------
# Save model
# -----------------------------------
model = {
    "features": X,
    "labels": y
}

with open(
    "sign_language_knn.pkl",
    "wb"
) as file:

    pickle.dump(model, file)


print()
print("==============================")
print("LEFT/RIGHT HAND KNN MODEL")
print("==============================")


for label in sorted(set(y)):

    count = y.count(label)

    print(
        label,
        "samples:",
        count
    )


print()
print("Model saved as:")
print("sign_language_knn.pkl")


# -----------------------------------
# KNN distance
# -----------------------------------
def distance(a, b):

    total = 0

    for i in range(len(a)):

        difference = a[i] - b[i]

        total += difference * difference

    return math.sqrt(total)


# -----------------------------------
# KNN prediction
# -----------------------------------
def predict(sample, k=7):

    distances = []

    for i in range(len(X)):

        d = distance(
            sample,
            X[i]
        )

        distances.append(
            (d, y[i])
        )

    distances.sort(
        key=lambda item: item[0]
    )

    nearest = distances[:k]

    votes = {}

    for d, label in nearest:

        votes[label] = (
            votes.get(label, 0) + 1
        )

    best_label = max(
        votes,
        key=votes.get
    )

    return best_label


# -----------------------------------
# Quick test
# -----------------------------------
correct = 0

test_count = min(
    100,
    len(X)
)

indices = random.sample(
    range(len(X)),
    test_count
)

for index in indices:

    prediction = predict(
        X[index]
    )

    if prediction == y[index]:

        correct += 1


accuracy = (
    correct /
    test_count
) * 100


print()
print(
    f"Quick KNN accuracy: {accuracy:.2f}%"
)