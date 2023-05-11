from Layers.Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
    
    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        output = error_tensor * (1 - self.activations**2)
        return output