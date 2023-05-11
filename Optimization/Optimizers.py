import sys 
import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = []

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer



class Sgd(Optimizer):
    def __init__(self,  learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor): 
        prev_weights = weight_tensor

        if self.regularizer:
            prev_weights = prev_weights - self.learning_rate * self.regularizer.calculate_gradient(prev_weights)

        new_weights = prev_weights - self.learning_rate*gradient_tensor
        return new_weights

            



class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.weightUpdate = 0.0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.weightUpdate = self.momentum_rate*self.weightUpdate - self.learning_rate*gradient_tensor

        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        output = weight_tensor + self.weightUpdate
        return output



class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.momentum_beta1 = 0.0
        self.momentum_beta2 = 0.0
        self.k = 0.0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        #for beta1
        self.momentum_beta1 = self.mu*self.momentum_beta1 + (1-self.mu)*gradient_tensor

        #for beta2
        self.momentum_beta2 = self.rho*self.momentum_beta2 + (1-self.rho)*gradient_tensor*gradient_tensor
        
        #bias correction
        self.k = self.k+1
        self.momentum_beta1_cap = self.momentum_beta1/(1-self.mu**self.k)
        self.momentum_beta2_cap = self.momentum_beta2/(1-self.rho**self.k)

        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        output = weight_tensor - self.learning_rate*(self.momentum_beta1_cap/(self.momentum_beta2_cap**0.5 + sys.float_info.epsilon))
        return output