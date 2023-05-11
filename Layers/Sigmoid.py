from Layers.Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, input_tensor):
        self.activations = 1/(1 + np.exp(-1*input_tensor))  
        return self.activations

    def backward(self, error_tensor):
        output = error_tensor * (self.activations * (1 - self.activations))
        return output