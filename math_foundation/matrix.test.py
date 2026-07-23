from matrix import Matrix


def test_matrix_class():
    print("Running Matrix Tests...")

    # 1. Init & Shape
    m1 = Matrix([[1, 2], [3, 4]])
    assert m1.shape == (2, 2), "Shape should be (2, 2)"
    print("✓ Init & Shape passed")

    # 2. Jagged Array Validation (Sad Path)
    try:
        bad_matrix = Matrix([[1, 2], [3]])  # Uneven rows
        assert False, "Should have raised ValueError for jagged array"
    except ValueError:
        pass
    print("✓ Jagged Array Validation passed")

    # 3. Addition
    m2 = Matrix([[5, 6], [7, 8]])
    m_add = m1 + m2
    assert m_add.shape == (2, 2)
    assert m_add._Matrix__data == [[6, 8], [10, 12]], "Addition math failed"
    print("✓ Addition passed")

    # 4. Subtraction
    m_sub = m2 - m1
    assert m_sub._Matrix__data == [[4, 4], [4, 4]], "Subtraction math failed"
    print("✓ Subtraction passed")

    # 5. Scalar Multiplication
    m_mul = m1 * 2
    assert m_mul._Matrix__data == [[2, 4], [6, 8]], "Scalar multiplication failed"
    print("✓ Scalar Multiplication passed")

    # 6. Reverse Scalar Multiplication (__rmul__)
    m_rmul = 3 * m1
    assert m_rmul._Matrix__data == [
        [3, 6],
        [9, 12],
    ], "Reverse scalar multiplication failed"
    print("✓ Reverse Scalar Multiplication passed")

    # 7. Transpose
    m_trans = m1.transpose()
    assert m_trans.shape == (2, 2)
    assert m_trans._Matrix__data == [[1, 3], [2, 4]], "Transpose failed"

    # Test transpose on a non-square matrix
    m_rect = Matrix([[1, 2, 3], [4, 5, 6]])  # Shape: 2x3
    m_rect_trans = m_rect.transpose()
    assert m_rect_trans.shape == (3, 2), "Rectangle transpose shape failed"
    assert m_rect_trans._Matrix__data == [
        [1, 4],
        [2, 5],
        [3, 6],
    ], "Rectangle transpose data failed"
    print("✓ Transpose passed")

    # 8. Dot Product (Square Matrices)
    # [[1, 2],    [[5, 6],    [[1*5 + 2*7, 1*6 + 2*8],    [[19, 22],
    #  [3, 4]]  .  [7, 8]]  =  [3*5 + 4*7, 3*6 + 4*8]]  =   [43, 50]]
    m_dot_square = m1.dot(m2)
    assert m_dot_square.shape == (2, 2), "Dot product output shape failed"
    assert m_dot_square._Matrix__data == [
        [19, 22],
        [43, 50],
    ], "Square dot product math failed"
    print("✓ Square Dot Product passed")

    # 9. Dot Product (Non-Square Matrices - The true test of your loops)
    # A (2x3) . B (3x2) = C (2x2)
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    B = Matrix([[7, 8], [9, 10], [11, 12]])

    m_dot_rect = A.dot(B)
    assert m_dot_rect.shape == (2, 2), "Rectangle dot product shape failed"

    # Row 0 of A dot Col 0 of B: (1*7) + (2*9) + (3*11) = 7 + 18 + 33 = 58
    # Row 0 of A dot Col 1 of B: (1*8) + (2*10) + (3*12) = 8 + 20 + 36 = 64
    # Row 1 of A dot Col 0 of B: (4*7) + (5*9) + (6*11) = 28 + 45 + 66 = 139
    # Row 1 of A dot Col 1 of B: (4*8) + (5*10) + (6*12) = 32 + 50 + 72 = 154
    assert m_dot_rect._Matrix__data == [
        [58, 64],
        [139, 154],
    ], "Rectangle dot product math failed"
    print("✓ Rectangle Dot Product passed")

    # ==========================================
    # SAD PATHS (Error Handling)
    # ==========================================

    # 10. Shape Mismatch on Addition
    m_2x3 = Matrix([[1, 2, 3], [4, 5, 6]])
    try:
        result = m1 + m_2x3  # 2x2 + 2x3
        assert False, "Should have raised ValueError for add shape mismatch"
    except ValueError:
        pass
    print("✓ Addition Shape Mismatch Error passed")

    # 11. Inner Dimension Mismatch on Dot Product
    # 2x2 . 2x3 -> inner dims are 2 and 2. Wait, that actually works! Let's do 2x2 . 3x2
    m_3x2 = Matrix([[1, 2], [3, 4], [5, 6]])
    try:
        result = m1.dot(m_3x2)  # 2x2 . 3x2 (cols of m1=2, rows of m_3x2=3) -> FAIL!
        assert (
            False
        ), "Should have raised ValueError for dot product inner dimension mismatch"
    except ValueError:
        pass
    print("✓ Dot Product Inner Dimension Error passed")

    print("\n" + "=" * 30)
    print("ALL MATRIX TESTS PASSED!")
    print("=" * 30)


# Run the tests
test_matrix_class()
