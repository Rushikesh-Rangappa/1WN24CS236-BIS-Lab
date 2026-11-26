import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. LOAD DATA
# =====================================================

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

num_features = X_train.shape[1]

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Total features:", num_features)


# =====================================================
# 2. CLASSIFIER AND FITNESS FUNCTION
# =====================================================

def create_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42)
    )

# Fitness = classification error + penalty for using many features
FEATURE_PENALTY = 0.01

def fitness(position):
    selected = np.where(position == 1)[0]

    # Avoid an empty feature subset
    if len(selected) == 0:
        return 1.0

    model = create_model()

    scores = cross_val_score(
        model,
        X_train[:, selected],
        y_train,
        cv=3,
        scoring="accuracy"
    )

    error = 1 - scores.mean()
    feature_ratio = len(selected) / num_features

    return error + FEATURE_PENALTY * feature_ratio


# =====================================================
# 3. BINARY GREY WOLF OPTIMIZER
# =====================================================

def binary_gwo(num_wolves=20, max_iterations=30):

    rng = np.random.default_rng(42)

    # Random binary feature subsets
    wolves = rng.integers(
        0, 2,
        size=(num_wolves, num_features)
    )

    # Ensure each wolf selects at least one feature
    for wolf in wolves:
        if not np.any(wolf):
            wolf[rng.integers(num_features)] = 1

    best_fitness_history = []

    alpha_position = None
    alpha_score = float("inf")

    for iteration in range(max_iterations):

        # Evaluate all wolves and rank them
        scores = np.array([fitness(wolf) for wolf in wolves])
        ranked_indices = np.argsort(scores)

        alpha = wolves[ranked_indices[0]].copy()
        beta = wolves[ranked_indices[1]].copy()
        delta = wolves[ranked_indices[2]].copy()

        if scores[ranked_indices[0]] < alpha_score:
            alpha_score = scores[ranked_indices[0]]
            alpha_position = alpha.copy()

        # a decreases from 2 to 0
        a = 2 - 2 * iteration / max(1, max_iterations - 1)

        new_wolves = []

        for wolf in wolves:

            candidate_positions = []

            # Update using alpha, beta, and delta
            for leader in [alpha, beta, delta]:

                r1 = rng.random(num_features)
                r2 = rng.random(num_features)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(C * leader - wolf)
                candidate = leader - A * D

                candidate_positions.append(candidate)

            # Average the three candidate positions
            continuous_position = np.mean(
                candidate_positions,
                axis=0
            )

            # Convert continuous values to binary using sigmoid
            probabilities = 1 / (
                1 + np.exp(-np.clip(continuous_position, -50, 50))
            )

            binary_position = (
                rng.random(num_features) < probabilities
            ).astype(int)

            if not np.any(binary_position):
                binary_position[rng.integers(num_features)] = 1

            new_wolves.append(binary_position)

        wolves = np.array(new_wolves)

        best_fitness_history.append(alpha_score)

        if (iteration + 1) % 5 == 0:
            selected_count = int(np.sum(alpha_position))
            print(
                f"Iteration {iteration + 1}/{max_iterations} | "
                f"Best fitness: {alpha_score:.4f} | "
                f"Selected features: {selected_count}"
            )

    return alpha_position, alpha_score, best_fitness_history


# =====================================================
# 4. RUN GWO
# =====================================================

best_features, best_fitness, history = binary_gwo(
    num_wolves=20,
    max_iterations=30
)

selected_indices = np.where(best_features == 1)[0]

print("\nSelected feature indices:")
print(selected_indices)

print("Number of selected features:", len(selected_indices))
print("Best fitness:", round(best_fitness, 4))


# =====================================================
# 5. TRAIN AND EVALUATE USING ALL FEATURES
# =====================================================

all_features_model = create_model()
all_features_model.fit(X_train, y_train)

all_predictions = all_features_model.predict(X_test)
all_accuracy = accuracy_score(y_test, all_predictions)


# =====================================================
# 6. TRAIN AND EVALUATE USING GWO FEATURES
# =====================================================

gwo_model = create_model()

gwo_model.fit(
    X_train[:, selected_indices],
    y_train
)

gwo_predictions = gwo_model.predict(
    X_test[:, selected_indices]
)

gwo_accuracy = accuracy_score(y_test, gwo_predictions)


# =====================================================
# 7. DISPLAY RESULTS
# =====================================================

print("\n========== RESULTS ==========")

print(f"Accuracy using all {num_features} features: "
      f"{all_accuracy * 100:.2f}%")

print(f"Accuracy using {len(selected_indices)} GWO features: "
      f"{gwo_accuracy * 100:.2f}%")

print(f"Features reduced: "
      f"{num_features - len(selected_indices)}")

print("\nClassification report using GWO-selected features:")
print(classification_report(y_test, gwo_predictions))


# =====================================================
# 8. PLOT CONVERGENCE
# =====================================================

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(history) + 1), history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Best fitness")
plt.title("Binary GWO Feature Selection Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()


# =====================================================
# 9. COMPARE ACCURACY
# =====================================================

plt.figure(figsize=(6, 4))
plt.bar(
    ["All features", "GWO-selected"],
    [all_accuracy * 100, gwo_accuracy * 100]
)
plt.ylabel("Test accuracy (%)")
plt.title("Classification Accuracy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()
