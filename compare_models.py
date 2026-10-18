import csv
import random
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# SETTINGS
# ==================================================

DATASET_DIR = Path("dataset")

SIGNS = [
    "HELLO",
    "NO",
    "STOP",
    "THANK YOU",
    "YES"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ==================================================
# NORMALIZE LANDMARKS
# Same method used by predict_knn.py
# ==================================================

def normalize_landmarks(values):

    landmarks = []

    for i in range(21):

        x = values[i * 3]
        y_value = values[i * 3 + 1]
        z = values[i * 3 + 2]

        landmarks.append([x, y_value, z])

    # Use wrist as origin
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


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset():

    features = []
    labels = []

    print("\nLoading dataset...\n")

    for sign in SIGNS:

        file_path = DATASET_DIR / f"{sign}.csv"

        if not file_path.exists():

            print(f"ERROR: File not found: {file_path}")
            raise SystemExit(1)

        count = 0

        with open(
            file_path,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.reader(file)

            for row in reader:

                if len(row) < 64:
                    continue

                try:

                    # First column is the label.
                    # Remaining 63 values are landmarks.
                    row_label = row[0].strip()

                    values = [
                        float(value)
                        for value in row[1:64]
                    ]

                except (ValueError, IndexError):
                    # Skip headers or invalid rows
                    continue

                if len(values) != 63:
                    continue

                normalized = normalize_landmarks(values)

                features.append(normalized)
                labels.append(row_label)

                count += 1

        print(f"{sign}: {count} samples")

    if not features:

        print("\nERROR: No valid samples were loaded.")
        print("Check the CSV files inside the dataset folder.")
        raise SystemExit(1)

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)

    print("\nTotal samples:", len(X))
    print("Feature count:", X.shape[1])

    return X, y


# ==================================================
# MIRROR TRAINING DATA ONLY
# ==================================================

def mirror_training_data(X_train, y_train):

    # Copy the original training features
    X_mirrored = X_train.copy()

    # Each landmark contains x, y, z.
    # Flip only the x coordinate.
    X_mirrored[:, 0::3] *= -1

    # Combine original and mirrored training data
    X_augmented = np.vstack([
        X_train,
        X_mirrored
    ])

    y_augmented = np.concatenate([
        y_train,
        y_train
    ])

    return X_augmented, y_augmented


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    X, y = load_dataset()

    # Split ORIGINAL samples before mirroring.
    # Test samples are never mirrored.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples before mirroring:", len(X_train))
    print("Testing samples:", len(X_test))

    # Augment training data only
    X_train, y_train = mirror_training_data(
        X_train,
        y_train
    )

    print("Training samples after mirroring:", len(X_train))

    # ==================================================
    # MODELS
    # ==================================================

    models = {

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="uniform",
            metric="euclidean"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced"
            )
        )
    }

    results = []

    # ==================================================
    # TRAIN AND EVALUATE
    # ==================================================

    for model_name, model in models.items():

        print("\n" + "=" * 55)
        print("MODEL:", model_name)
        print("=" * 55)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        ) * 100

        results.append((model_name, accuracy))

        print(f"\nAccuracy: {accuracy:.2f}%")

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions,
                labels=SIGNS,
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print("Class order:", SIGNS)

        print(
            confusion_matrix(
                y_test,
                predictions,
                labels=SIGNS
            )
        )

    # ==================================================
    # FINAL COMPARISON
    # ==================================================

    print("\n")
    print("=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for rank, (name, accuracy) in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. {name}: {accuracy:.2f}%"
        )

    print("\nEvaluation completed.")
    print("Your existing model files were not modified.")


if __name__ == "__main__":
    main()