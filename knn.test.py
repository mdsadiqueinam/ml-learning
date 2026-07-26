from knn_classifier import KNNClassifier
from math_foundation.matrix import Matrix
from math_foundation.vector import Vector

# from knn import KNNClassifier  # Adjust import as needed


def test_knn():
    print("Running KNN Tests...")

    # Features: [Weight in grams, Redness score 0-10]
    X_train = Matrix(
        [
            [150, 9],  # Heavy, very red (Apple)
            [160, 8],  # Heavy, red (Apple)
            [170, 9],  # Heavy, very red (Apple)
            [120, 2],  # Light, not red (Banana)
            [130, 3],  # Light, slightly red (Banana)
            [110, 1],  # Light, green (Banana)
        ]
    )
    # 0 = Apple, 1 = Banana
    y_train = Vector([0, 0, 0, 1, 1, 1])

    # ---------------------------------------------------------
    # TEST 1: Obvious Banana
    # ---------------------------------------------------------
    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)

    X_test_banana = Matrix([[140, 1]])  # Light, not red
    pred_banana = model.predict(X_test_banana)

    assert (
        pred_banana.components[0] == 1
    ), f"Should be Banana (1), got {pred_banana.components[0]}"
    print("✓ Test 1 Passed: Correctly classified an obvious Banana")

    # ---------------------------------------------------------
    # TEST 2: Obvious Apple
    # ---------------------------------------------------------
    X_test_apple = Matrix([[155, 10]])  # Heavy, very red
    pred_apple = model.predict(X_test_apple)

    assert (
        pred_apple.components[0] == 0
    ), f"Should be Apple (0), got {pred_apple.components[0]}"
    print("✓ Test 2 Passed: Correctly classified an obvious Apple")

    # ---------------------------------------------------------
    # TEST 3: Batch Prediction (Multiple mystery fruits at once)
    # ---------------------------------------------------------
    X_test_batch = Matrix(
        [
            [115, 2],  # Banana
            [165, 9],  # Apple
            [125, 4],  # Banana (closer to banana cluster)
        ]
    )
    pred_batch = model.predict(X_test_batch)

    assert pred_batch.components == [
        1,
        0,
        1,
    ], f"Batch failed, got {pred_batch.components}"
    print("✓ Test 3 Passed: Correctly classified a batch of 3 fruits")

    # ---------------------------------------------------------
    # TEST 4: Odd vs Even K (The Tie-Breaker rule)
    # ---------------------------------------------------------
    # Let's put a point exactly halfway between an apple and a banana
    X_test_tie = Matrix([[140, 5]])

    # If k=2 (even), it might find 1 apple and 1 banana. Counter will just pick one.
    model_even = KNNClassifier(k=2)
    model_even.fit(X_train, y_train)
    pred_even = model_even.predict(X_test_tie)

    # If k=3 (odd), it forces a majority. It should lean banana because 140 is closer to 120/130.
    model_odd = KNNClassifier(k=3)
    model_odd.fit(X_train, y_train)
    pred_odd = model_odd.predict(X_test_tie)

    # We don't assert the exact label for the tie, just that it doesn't crash
    # and returns a valid 0 or 1. (This teaches you why we almost always use odd numbers for K!)
    assert pred_even.components[0] in [0, 1], "Even K failed to return a valid label"
    assert pred_odd.components[0] in [0, 1], "Odd K failed to return a valid label"
    print("✓ Test 4 Passed: Handled edge-case data points without crashing")

    print("\n" + "=" * 30)
    print("ALL KNN TESTS PASSED!")
    print("=" * 30)


test_knn()
