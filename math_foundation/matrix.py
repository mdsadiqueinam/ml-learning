from __future__ import division
from functools import reduce
from typing import Self
from collections.abc import Callable


class Matrix:

    def __init__(self, data: list[list[float]]):
        # validate shape
        expected_cols = len(data[0])
        if not all(list(map(lambda row: len(row) == expected_cols, data))):
            raise ValueError("Matrix shape uneven")

        self.__data = data

    @property
    def data(self) -> list[list[float]]:
        return self.__data

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.__data), len(self.__data[0]))

    def __validate_shape(self, other: Self):
        if self.shape != other.shape:
            raise ValueError("Matrix shape mismatch")

    def __process_math(
        self, other: Self, operation: Callable[[float, float], float]
    ) -> list[list[float]]:
        new_data = []
        for i in range(len(self.__data)):
            new_row = []
            for j in range(len(self.__data[0])):
                new_row.append(operation(self.__data[i][j], other.__data[i][j]))
            new_data.append(new_row)
        return new_data

    def __add__(self, other: Self) -> Self:
        self.__validate_shape(other)
        new_data = self.__process_math(other, lambda comp1, comp2: comp1 + comp2)
        return Matrix(new_data)

    def __sub__(self, other: Self) -> Self:
        self.__validate_shape(other)
        new_data = self.__process_math(other, lambda comp1, comp2: comp1 - comp2)
        return Matrix(new_data)

    def __mul__(self, scalar: float | int) -> Self:
        return Matrix(
            list(map(lambda row: list(map(lambda col: col * scalar, row)), self.__data))
        )

    def __rmul__(self, scalar: float | int):
        return self.__mul__(scalar)

    def transpose(self) -> Self:
        new_data = []
        for j in range(len(self.__data[0])):
            new_row = []
            for row in self.__data:
                new_row.append(row[j])
            new_data.append(new_row)
        return Matrix(new_data)

    def dot(self, other: Self) -> Self:
        n, p = other.shape
        if self.shape[1] != n:
            raise ValueError("Matrix shape is not suitable for dot product")
        new_data = []

        for row in self.__data:
            new_row = []
            for i in range(p):
                summation = 0
                for j in range(n):
                    summation += row[j] * other.__data[j][i]
                new_row.append(summation)
            new_data.append(new_row)

        return Matrix(new_data)
