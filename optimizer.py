from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def step(self, w: float, b: float, dw: float, db: float) -> tuple[float, float]:
        pass


class VanillaGD(Optimizer):

    def __init__(self, learning_rate: float):
        super().__init__()
        self.lr = learning_rate

    def step(self, w, b, dw, db):
        w = w - (self.lr * dw)
        b = b - (self.lr * db)
        return (w, b)


class MomentumGD(Optimizer):

    def __init__(self, learning_rate: float, momentun: float):
        super().__init__()
        self.lr = learning_rate
        self.momentun = momentun
        self.v_w = 0.0
        self.v_b = 0.0

    def step(self, w, b, dw, db):
        self.v_w = (self.momentun * self.v_w) + dw
        w = w - (self.lr * self.v_w)

        self.v_b = (self.momentun * self.v_b) + db
        b = b - (self.lr * self.v_b)

        return (w, b)
