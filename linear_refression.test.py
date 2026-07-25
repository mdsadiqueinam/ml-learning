from linear_regression import LinearRegression
from math_foundation.vector import Vector

# Adjust import path if needed


def test_linear_regression():
    print("Running Linear Regression Tests...")

    # ---------------------------------------------------------
    # TEST 1: Pre-training predictions (Sanity Check)
    # ---------------------------------------------------------
    model = LinearRegression()
    X_dummy = Vector([10, 20, 30])

    # Since w=0 and b=0, predictions should be all zeros
    preds = model.predict(X_dummy)
    assert preds.components == [0.0, 0.0, 0.0], "Initial predictions should be 0"
    print("✓ Test 1 Passed: Initial predictions are zero")

    # ---------------------------------------------------------
    # TEST 2: Pre-training MSE (Sanity Check)
    # ---------------------------------------------------------
    Y_dummy = Vector([5, 10, 15])
    initial_mse = model.mse(X_dummy, Y_dummy)

    # Errors are [5, 10, 15]. Squared: [25, 100, 225]. Mean: 350/3 = 116.666...
    expected_mse = (5**2 + 10**2 + 15**2) / 3
    assert abs(initial_mse - expected_mse) < 1e-9, "Initial MSE calculation is wrong"
    print("✓ Test 2 Passed: Initial MSE calculated correctly")

    # ---------------------------------------------------------
    # TEST 3: Perfect Linear Data (y = 2x + 0)
    # ---------------------------------------------------------
    model_perfect = LinearRegression()
    X_perfect = Vector([1, 2, 3, 4])
    Y_perfect = Vector([2, 4, 6, 8])  # Exact slope of 2, no bias

    model_perfect.fit(X_perfect, Y_perfect, learning_rate=0.1, epochs=100)

    assert (
        abs(model_perfect.w - 2.0) < 0.01
    ), f"Weight should be ~2.0, got {model_perfect.w}"
    assert (
        abs(model_perfect.b - 0.0) < 0.05
    ), f"Bias should be ~0.0, got {model_perfect.b}"
    print("✓ Test 3 Passed: Learned perfect linear relationship (y=2x)")

    # ---------------------------------------------------------
    # TEST 4: Perfect Linear Data WITH Bias (y = 3x + 5)
    # ---------------------------------------------------------
    model_bias = LinearRegression()
    X_bias = Vector([0, 1, 2, 3])
    Y_bias = Vector([5, 8, 11, 14])  # Slope 3, Bias 5

    model_bias.fit(X_bias, Y_bias, learning_rate=0.1, epochs=100)

    assert abs(model_bias.w - 3.0) < 0.01, f"Weight should be ~3.0, got {model_bias.w}"
    assert abs(model_bias.b - 5.0) < 0.01, f"Bias should be ~5.0, got {model_bias.b}"
    print("✓ Test 4 Passed: Learned perfect linear relationship with bias (y=3x+5)")

    # ---------------------------------------------------------
    # TEST 5: Noisy Real-World Data (The House Pricing Example)
    # ---------------------------------------------------------
    model_houses = LinearRegression()
    # Sq ft
    X_houses = Vector([1000, 1500, 2000, 2500])
    # Prices (Not perfectly linear, has some noise)
    Y_houses = Vector([205000, 295000, 405000, 495000])
    # True underlying rule is roughly: price = 200*sqft + 5000

    # WARNING: Learning rate MUST be tiny because X values are huge!
    initial_house_mse = model_houses.mse(X_houses, Y_houses)
    model_houses.fit(X_houses, Y_houses, learning_rate=0.0000001, epochs=1000)
    final_house_mse = model_houses.mse(X_houses, Y_houses)

    assert final_house_mse < initial_house_mse, "MSE should go down after training!"
    assert (
        abs(model_houses.w - 200.0) < 5.0
    ), f"Weight should be ~200.0, got {model_houses.w}"
    assert (
        abs(model_houses.b - 5000.0) < 5000.0
    ), f"Bias should be ~5000.0, got {model_houses.b}"  # Bias is harder to nail perfectly with noise
    print("✓ Test 5 Passed: Learned from noisy real-world data")

    # ---------------------------------------------------------
    # TEST 6: Final Prediction Check
    # ---------------------------------------------------------
    # Using the house model, predict a 3000 sq ft house
    prediction_3000 = model_houses.predict(Vector([3000]))
    expected_3000 = (200 * 3000) + 5000

    assert (
        abs(prediction_3000.components[0] - expected_3000) < 15000
    ), "Prediction for new data is way off"
    print("✓ Test 6 Passed: Can predict on unseen data")

    print("\n" + "=" * 40)
    print("ALL LINEAR REGRESSION TESTS PASSED!")
    print("=" * 40)


test_linear_regression()
