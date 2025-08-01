import numpy as np

class SGD:
    """確率的勾配降下法（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.01):
        self.lr =lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    """Momentum SGD"""
    def __init__(self, lr=0.01, momemtum=0.9):
        self.lr =lr
        self.momemtum = momemtum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momemtum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """AdaGrad"""
    def __init__(self, lr=0.01):
        self.lr =lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)



