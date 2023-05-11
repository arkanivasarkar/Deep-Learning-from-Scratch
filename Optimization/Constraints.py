import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        output = self.alpha * weights
        return output

    def norm(self, weights):
        output = self.alpha * np.sum(weights**2)
        return output


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def calculate_gradient(self, weights):
        output = self.alpha * np.sign(weights)
        return output

    def norm(self, weights):
        output = self.alpha * np.sum(np.abs(weights))
        return output


