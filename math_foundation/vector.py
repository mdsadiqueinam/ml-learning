from __future__ import division
from typing import Self
from math import sqrt


class Vector:

    def __init__(self, components: list[float]):
        self.__components = components

    @property
    def components(self) -> list[float]:
        return self.__components.copy()

    @property
    def dimension(self) -> int:
        return len(self.__components)

    def __validate_dimension(self, other: Self):
        if other.dimension != self.dimension:
            raise ValueError("Vector dimenion mismatch")

    def __validate_type(self, other: Self):
        if type(self) != type(other):
            raise TypeError("Other type is not vector")

    def __add__(self, other: Self) -> Self:
        self.__validate_type(other)
        self.__validate_dimension(other)

        new_components = []
        for i in range(self.dimension):
            new_components.append(self.__components[i] + other.__components[i])

        return Vector(new_components)

    def __sub__(self, other: Self) -> Self:
        self.__validate_type(other)
        self.__validate_dimension(other)

        new_components = []
        for i in range(self.dimension):
            new_components.append(self.__components[i] - other.__components[i])

        return Vector(new_components)

    def __mul__(self, scalar: float | int) -> Self:
        return Vector(list(map(lambda comp: comp * scalar, self.__components)))

    def __rmul__(self, scalar: float | int) -> Self:
        return self.__mul__(scalar)

    def dot(self, other: Self) -> float:
        self.__validate_type(other)
        self.__validate_dimension(other)
        product = 0
        for i in range(self.dimension):
            product += self.__components[i] * other.__components[i]

        return product

    def magnitude(self) -> float:
        # Sum of squares is equal to dot product of itself.
        return sqrt(self.dot(self))

    def norm(self) -> Self:
        magnitube = self.magnitude()

        if magnitube == 0:
            return self

        new_comp = list(map(lambda comp: comp / magnitube, self.__components))
        return Vector(new_comp)

    def distance(self, other: Self) -> float:
        self.__validate_type(other)
        self.__validate_dimension(other)

        displacement_vector = self - other
        return displacement_vector.magnitude()
