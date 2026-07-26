from math_foundation.vector import Vector
from optimizer import Optimizer


class LinearRegression:

    def __init__(self, optimizer: Optimizer):
        # weight
        self.w = 0.0
        # bias
        self.b = 0.0

        self.optimizer = optimizer

    def predict(self, X: Vector):
        # by the formula y = mx + b here m = w
        return Vector(list(map(lambda x: (self.w * x) + self.b, X.components)))

    def mse(self, X: Vector, Y: Vector) -> float:
        prediction = self.predict(X)
        errors = Y - prediction
        return errors.dot(errors) / errors.dimension

    def fit(self, X: Vector, Y: Vector, epochs: int):
        for i in range(epochs):
            predictions = self.predict(X)
            errors = predictions - Y

            dw = errors.dot(X) * (2 / errors.dimension)

            db = sum(errors.components) * (2 / errors.dimension)
            self.w, self.b = self.optimizer.step(self.w, self.b, dw, db)
