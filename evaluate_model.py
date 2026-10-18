import csv
import math
import random
from collections import Counter


# =========================================================
# SETTINGS
# =========================================================

TEST_RATIO = 0.20
K = 7

DATASET_FOLDER = "dataset"


# =========================================================
# NORMALIZE LANDMARKS
# =========================================================

def normalize_landmarks(values):

    landmarks = []

    for i in range(21):

        x = values[i * 3]
        y = values[i * 3 + 1]
        z = values[i * 3 + 2]

        landmarks.append([
            x, y, z
        ])

    # Wrist as origin
    wrist = landmarks[0]

    for point in landmarks:

        point[0] -= wrist[0]
        point[1] -= wrist[1]
        point[2] -= wrist[2]

    # Hand width
    min_x = min(
        p[0] for p in landmarks
    )

    max_x = max(
        p[0] for p in landmarks
    )

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
# MIRROR SAMPLE
# =========================================================

def mirror_sample(features):

    mirrored = []

    for i in range(0, len(features), 3):

        x = features[i]
        y = features[i + 1]
        z = features[i + 2]

        mirrored.extend([
            -x,
            y,
            z
        ])

    return mirrored


# =========================================================
# LOAD DATA
# =========================================================

all_data = []


files = [
    "HELLO.csv",
    "NO.csv",
    "STOP.csv",
    "THANK YOU.csv",
    "YES.csv"
]


for filename in files:

    path = (
        DATASET_FOLDER
        + "/"
        + filename
    )

    with open(
        path,
        "r",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) != 64:
                continue

            try:

                label = row[0]

                values = [
                    float(x)
                    for x in row[1:]
                ]

                features = normalize_landmarks(
                    values
                )

                all_data.append(
                    (features, label)
                )

            except:

                continue


print()
print("Total original samples:", len(all_data))


# =========================================================
# SHUFFLE DATA
# =========================================================

random.seed(42)

random.shuffle(all_data)


# =========================================================
# STRATIFIED SPLIT
# =========================================================

classes = sorted(
    set(label for features, label in all_data)
)

train_data = []
test_data = []


for label in classes:

    class_samples = [
        item
        for item in all_data
        if item[1] == label
    ]

    random.shuffle(class_samples)

    test_count = max(
        1,
        int(
            len(class_samples)
            * TEST_RATIO
        )
    )

    test_samples = class_samples[
        :test_count
    ]

    train_samples = class_samples[
        test_count:
    ]

    train_data.extend(
        train_samples
    )

    test_data.extend(
        test_samples
    )


print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))


# =========================================================
# ADD MIRRORED TRAINING DATA
# =========================================================

training_features = []
training_labels = []


for features, label in train_data:

    # Original
    training_features.append(
        features
    )

    training_labels.append(
        label
    )

    # Mirrored
    training_features.append(
        mirror_sample(features)
    )

    training_labels.append(
        label
    )


print(
    "Training samples after mirroring:",
    len(training_features)
)


# =========================================================
# DISTANCE
# =========================================================

def distance(a, b):

    total = 0

    for i in range(len(a)):

        difference = (
            a[i] - b[i]
        )

        total += (
            difference *
            difference
        )

    return math.sqrt(total)


# =========================================================
# KNN PREDICTION
# =========================================================

def predict(sample):

    distances = []

    for i in range(
        len(training_features)
    ):

        d = distance(
            sample,
            training_features[i]
        )

        distances.append(
            (d, training_labels[i])
        )


    distances.sort(
        key=lambda x: x[0]
    )


    nearest = distances[:K]


    votes = Counter(
        label
        for distance_value, label
        in nearest
    )


    prediction = votes.most_common(
        1
    )[0][0]


    return prediction


# =========================================================
# TEST MODEL
# =========================================================

actual = []
predicted = []


print()
print("Testing model...")
print()


for index, (
    features,
    label
) in enumerate(test_data):

    prediction = predict(
        features
    )

    actual.append(label)
    predicted.append(prediction)

    print(
        str(index + 1)
        + "/"
        + str(len(test_data)),
        "Actual:",
        label,
        "| Predicted:",
        prediction
    )


# =========================================================
# ACCURACY
# =========================================================

correct = 0

for i in range(len(actual)):

    if actual[i] == predicted[i]:

        correct += 1


accuracy = (
    correct /
    len(actual)
) * 100


print()
print("=" * 60)

print(
    "ACCURACY:",
    round(accuracy, 2),
    "%"
)

print("=" * 60)


# =========================================================
# CONFUSION MATRIX
# =========================================================

matrix = {}

for actual_label in classes:

    matrix[actual_label] = {}

    for predicted_label in classes:

        matrix[actual_label][
            predicted_label
        ] = 0


for i in range(len(actual)):

    matrix[
        actual[i]
    ][
        predicted[i]
    ] += 1


print()
print("CONFUSION MATRIX")
print()

print(
    "Actual \\ Predicted",
    end="\t"
)

for label in classes:

    print(
        label,
        end="\t"
    )

print()


for actual_label in classes:

    print(
        actual_label,
        end="\t\t"
    )

    for predicted_label in classes:

        print(
            matrix[
                actual_label
            ][
                predicted_label
            ],
            end="\t"
        )

    print()


# =========================================================
# PRECISION / RECALL / F1
# =========================================================

print()
print("=" * 60)
print("PER-CLASS PERFORMANCE")
print("=" * 60)


for label in classes:

    TP = matrix[label][label]

    FP = 0
    FN = 0

    for other in classes:

        if other != label:

            FP += matrix[
                other
            ][
                label
            ]

            FN += matrix[
                label
            ][
                other
            ]


    # Precision
    if TP + FP == 0:

        precision = 0

    else:

        precision = (
            TP /
            (TP + FP)
        )


    # Recall
    if TP + FN == 0:

        recall = 0

    else:

        recall = (
            TP /
            (TP + FN)
        )


    # F1
    if precision + recall == 0:

        f1 = 0

    else:

        f1 = (
            2 *
            precision *
            recall /
            (precision + recall)
        )


    print()

    print(
        label
    )

    print(
        "  Precision:",
        round(
            precision * 100,
            2
        ),
        "%"
    )

    print(
        "  Recall:",
        round(
            recall * 100,
            2
        ),
        "%"
    )

    print(
        "  F1-score:",
        round(
            f1 * 100,
            2
        ),
        "%"
    )


print()
print("=" * 60)
print("Evaluation completed!")
print("=" * 60)