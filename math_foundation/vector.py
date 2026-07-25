from __future__ import division
from typing import Self
from math import sqrt


class Vector:
    """
    Represents a single column of data in a spreadsheet (e.g., a list of all house prices,
    or a single row of features for one house).

    In ML, we rarely use this for 3D geometric arrows. We use it to hold lists of numbers
    so we can do math on thousands of rows at the exact same time.
    """

    def __init__(self, components: list[float]):
        """
        Args:
            components: A list of numbers representing data points or features.

        Example:
            >>> # A column of 3 house prices in thousands
            >>> prices = Vector([200, 300, 250])
        """
        self.__components = components

    @property
    def components(self) -> list[float]:
        """
        Returns a safe copy of the data so external code can't accidentally mutate it.

        Example:
            >>> v = Vector([1, 2])
            >>> v.components
            [1, 2]
        """
        return self.__components.copy()

    @property
    def dimension(self) -> int:
        """
        Returns the number of items in the list (e.g., number of data samples,
        or number of features for one sample).

        Example:
            >>> v = Vector([10, 20, 30, 40])
            >>> v.dimension
            4
        """
        return len(self.__components)

    def __validate_dimension(self, other: Self):
        """Helper to ensure we aren't comparing different sized datasets."""
        if other.dimension != self.dimension:
            raise ValueError("Vector dimension mismatch")

    def __validate_type(self, other: Self):
        """Helper to ensure we are only doing math with other Vectors."""
        if type(self) != type(other):
            raise TypeError("Other type is not vector")

    def __add__(self, other: Self) -> Self:
        """
        Adds two columns of data together, row by row.

        Example:
            >>> # Add base house price to renovation cost
            >>> base_prices = Vector([100, 200])
            >>> reno_costs = Vector([20, 50])
            >>> total = base_prices + reno_costs
            >>> total.components
            [120, 250]
        """
        self.__validate_type(other)
        self.__validate_dimension(other)

        new_components = []
        for i in range(self.dimension):
            new_components.append(self.__components[i] + other.__components[i])

        return Vector(new_components)

    def __sub__(self, other: Self) -> Self:
        """
        Subtracts two columns of data, row by row. (Crucial for calculating errors!)

        Example:
            >>> # True price minus Predicted price = Error
            >>> true_price = Vector([300, 400])
            >>> predicted_price = Vector([310, 380])
            >>> error = true_price - predicted_price
            >>> error.components
            [-10, 20]
        """
        self.__validate_type(other)
        self.__validate_dimension(other)

        new_components = []
        for i in range(self.dimension):
            new_components.append(self.__components[i] - other.__components[i])

        return Vector(new_components)

    def __mul__(self, scalar: float | int) -> Self:
        """
        Scales every data point in the column by a single number.

        Example:
            >>> # Convert prices from thousands to actual dollars
            >>> prices_k = Vector([100, 200])
            >>> prices_actual = prices_k * 1000
            >>> prices_actual.components
            [100000, 200000]
        """
        return Vector(list(map(lambda comp: comp * scalar, self.__components)))

    def __rmul__(self, scalar: float | int) -> Self:
        """
        Allows multiplying from the left side: 1000 * vector

        Example:
            >>> tax_rate = 1.1
            >>> prices = Vector([100, 200])
            >>> new_prices = tax_rate * prices
        """
        return self.__mul__(scalar)

    def dot(self, other: Self) -> float:
        """
        The "Weighted Sum" engine of Machine Learning. Multiplies pairs of numbers
        from two lists and adds them all together. Used to calculate predictions
        and similarities.

        Example:
            >>> # Predict house price: (sq_ft * weight_sqft) + (bedrooms * weight_beds)
            >>> house_features = Vector([2000, 3])
            >>> model_weights = Vector([150, 10000])
            >>> predicted_price = house_features.dot(model_weights)
            >>> predicted_price
            330000.0  # (2000*150) + (3*10000)
        """
        self.__validate_type(other)
        self.__validate_dimension(other)
        product = 0
        for i in range(self.dimension):
            product += self.__components[i] * other.__components[i]

        return product

    def magnitude(self) -> float:
        """
        The total "size" or "length" of the data column. Calculated by doing
        a dot product of the column with itself, then taking the square root.

        Example:
            >>> error_vector = Vector([3, 4])
            >>> error_vector.magnitude()
            5.0
        """
        return sqrt(self.dot(self))

    def norm(self) -> Self:
        """
        Shrinks or stretches the data so its total magnitude is exactly 1.
        Used in ML to prevent features with big numbers (like salary) from
        drowning out features with small numbers (like number of kids).

        Example:
            >>> raw_data = Vector([3, 4]) # magnitude is 5
            >>> normalized = raw_data.norm()
            >>> normalized.components
            [0.6, 0.8] # magnitude is now exactly 1.0
        """
        magnitube = self.magnitude()

        if magnitube == 0:
            return self

        new_comp = list(map(lambda comp: comp / magnitube, self.__components))
        return Vector(new_comp)

    def distance(self, other: Self) -> Self:
        """
        Calculates the straight-line difference between two data points.
        Used heavily in algorithms like K-Nearest Neighbors to find "similar" items.

        Example:
            >>> house_a_features = Vector([2000, 3])
            >>> house_b_features = Vector([2050, 3])
            >>> house_a_features.distance(house_b_features)
            50.0  # They only differ by 50 sq ft
        """
        self.__validate_type(other)
        self.__validate_dimension(other)

        displacement_vector = self - other
        return displacement_vector.magnitude()
