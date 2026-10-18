import csv
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# SETTINGS
# ==========================================

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


# ==========================================
# NORMALIZE HAND LANDMARKS
# Same method as predict_knn.py
# ==========================================

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


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    features = []
    labels = []

    print("\nLoading dataset...\n")

    for sign in SIGNS:

        file_path = DATASET_DIR / f"{sign}.csv"

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset file not found: {file_path}"
            )

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

                    label = row[0].strip()

                    values = [
                        float(value)
                        for value in row[1:64]
                    ]

                except (ValueError, IndexError):
                    # Skip invalid rows or headers
                    continue

                if len(values) != 63:
                    continue

                features.append(
                    normalize_landmarks(values)
                )

                labels.append(label)

                count += 1

        print(f"{sign}: {count} samples")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)

    if len(X) == 0:
        raise ValueError("No valid samples found.")

    print("\nTotal samples:", len(X))

    return X, y


# ==========================================
# MIRROR TRAINING DATA ONLY
# ==========================================

def augment_training_data(X_train, y_train):

    X_mirrored = X_train.copy()

    # Flip the x coordinate of every landmark
    X_mirrored[:, 0::3] *= -1

    X_augmented = np.vstack([
        X_train,
        X_mirrored
    ])

    y_augmented = np.concatenate([
        y_train,
        y_train
    ])

    return X_augmented, y_augmented


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    X, y = load_dataset()

    # Split original data before augmentation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Mirror training data only
    X_train, y_train = augment_training_data(
        X_train,
        y_train
    )

    print("\nTraining samples after mirroring:", len(X_train))
    print("Testing samples:", len(X_test))

    # ======================================
    # KNN CONFIGURATIONS
    # ======================================

    configurations = [
        (3, "uniform"),
        (5, "uniform"),
        (7, "uniform"),
        (9, "uniform"),
        (3, "distance"),
        (5, "distance"),
        (7, "distance"),
        (9, "distance")
    ]

    results = []

    print("\n")
    print("=" * 65)
    print("KNN HYPERPARAMETER COMPARISON")
    print("=" * 65)

    # ======================================
    # TRAIN AND TEST EACH CONFIGURATION
    # ======================================

    for k, weighting in configurations:

        model = KNeighborsClassifier(
            n_neighbors=k,
            weights=weighting,
            metric="euclidean"
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        ) * 100

        results.append({
            "k": k,
            "weighting": weighting,
            "accuracy": accuracy,
            "model": model,
            "predictions": predictions
        })

        print(
            f"K={k:<2} | "
            f"Weights={weighting:<8} | "
            f"Accuracy={accuracy:.2f}%"
        )

    # ======================================
    # SORT RESULTS
    # ======================================

    results.sort(
        key=lambda result: result["accuracy"],
        reverse=True
    )

    print("\n")
    print("=" * 65)
    print("RANKING")
    print("=" * 65)

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"K={result['k']}, "
            f"Weights={result['weighting']}, "
            f"Accuracy={result['accuracy']:.2f}%"
        )

    # ======================================
    # BEST CONFIGURATION
    # ======================================

    best = results[0]

    print("\n")
    print("=" * 65)
    print("BEST CONFIGURATION ON THIS TEST SPLIT")
    print("=" * 65)

    print("Number of neighbors:", best["k"])
    print("Weighting:", best["weighting"])
    print(f"Accuracy: {best['accuracy']:.2f}%")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            best["predictions"],
            labels=SIGNS,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print("Class order:", SIGNS)

    print(
        confusion_matrix(
            y_test,
            best["predictions"],
            labels=SIGNS
        )
    )

    print("\nExperiment completed.")
    print("No existing model files were changed.")


if __name__ == "__main__":
    main()