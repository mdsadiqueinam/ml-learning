from vector import Vector


def test_vector_class():
    print("Running Vector Tests...")

    # 1. Init & Properties
    v1 = Vector([1, 2, 3])
    assert v1.dimension == 3, "Dimension should be 3"
    assert v1.components == [1, 2, 3], "Components should match init list"
    print("✓ Init & Properties passed")

    # 2. Addition
    v2 = Vector([4, 5, 6])
    v_add = v1 + v2
    assert v_add.components == [5, 7, 9], "Addition math failed"
    assert isinstance(v_add, Vector), "Addition must return a Vector object"
    print("✓ Addition passed")

    # 3. Subtraction
    v_sub = v2 - v1
    assert v_sub.components == [3, 3, 3], "Subtraction math failed"
    print("✓ Subtraction passed")

    # 4. Scalar Multiplication
    v_mul = v1 * 3
    assert v_mul.components == [3, 6, 9], "Scalar multiplication failed"
    assert isinstance(v_mul, Vector), "Multiplication must return a Vector object"

    v_mul_neg = v1 * -1
    assert v_mul_neg.components == [-1, -2, -3], "Negative scalar multiplication failed"
    print("✓ Scalar Multiplication passed")

    # 5. Dot Product
    # [1*4 + 2*5 + 3*6] = 4 + 10 + 18 = 32
    assert v1.dot(v2) == 32, "Dot product failed"

    # Orthogonal vectors (90 degrees) should have a dot product of 0
    v_orth_1 = Vector([1, 0])
    v_orth_2 = Vector([0, 1])
    assert v_orth_1.dot(v_orth_2) == 0, "Orthogonal dot product should be 0"
    print("✓ Dot Product passed")

    # 6. Magnitude
    # 3^2 + 4^2 = 9 + 16 = 25. sqrt(25) = 5.0
    v_mag = Vector([3, 4])
    assert v_mag.magnitude() == 5.0, "Magnitude failed"

    v_zero = Vector([0, 0, 0])
    assert v_zero.magnitude() == 0.0, "Zero vector magnitude should be 0"
    print("✓ Magnitude passed")

    # 7. Normalization
    # Magnitude is 5, so [3/5, 4/5] = [0.6, 0.8]
    v_norm = v_mag.norm()
    assert v_norm.components[0] == 0.6, "Norm X failed"
    assert v_norm.components[1] == 0.8, "Norm Y failed"

    # The magnitude of a normalized vector MUST be exactly 1
    assert abs(v_norm.magnitude() - 1.0) < 1e-9, "Normalized vector magnitude must be 1"
    print("✓ Normalization passed")

    # 8. Normalization Edge Case (Zero Vector)
    # Should not divide by zero, should just return a zero vector
    v_zero_norm = v_zero.norm()
    assert v_zero_norm.components == [0, 0, 0], "Zero vector norm failed"
    print("✓ Zero Vector Normalization passed")

    # 9. Distance
    v_dist_1 = Vector([0, 0])
    v_dist_2 = Vector([3, 4])
    # Distance is just the magnitude of the difference vector (which is 5)
    assert v_dist_1.distance(v_dist_2) == 5.0, "Distance failed"
    print("✓ Distance passed")

    # ==========================================
    # SAD PATHS (Error Handling)
    # ==========================================

    # 10. Dimension Mismatch
    v_short = Vector([1, 2])
    try:
        result = v1 + v_short
        assert False, "Should have raised ValueError for dimension mismatch"
    except ValueError:
        pass
    print("✓ Dimension Mismatch Error passed")

    # 11. Type Mismatch
    try:
        result = v1 + [4, 5, 6]  # Passing a raw list instead of a Vector
        assert False, "Should have raised TypeError for type mismatch"
    except TypeError:
        pass
    print("✓ Type Mismatch Error passed")

    print("\n" + "=" * 30)
    print("ALL VECTOR TESTS PASSED!")
    print("=" * 30)


# Run the tests
test_vector_class()
