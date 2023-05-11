from Layers.Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output = input_tensor*np.greater(input_tensor,0)
        return output 

    def backward(self, error_tensor):
        output = error_tensor * np.greater(self.input_tensor,0)
        return output

