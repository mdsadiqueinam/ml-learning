from collections import Counter

from math_foundation.matrix import Matrix
from math_foundation.vector import Vector


class KNNClassifier:

    def __init__(self, k: int):
        self.k = k

    def fit(self, X: Matrix, y: Vector):
        self.X = X
        self.y = y

    def _get_distances(self, new_point: Vector) -> list[float]:
        return list(map(lambda x: new_point.distance(Vector(x)), self.X.data))

    def predict(self, X_new: Matrix) -> Vector:
        labels = []
        for row in X_new.data:
            distance_pair = sorted(
                enumerate(self._get_distances(Vector(row))), key=lambda d: d[1]
            )
            top_items = Counter(
                list(map(lambda x: self.y.get(x[0]), distance_pair[: self.k]))
            )
            labels.append(top_items.most_common(1)[0][0])

        return Vector(labels)
