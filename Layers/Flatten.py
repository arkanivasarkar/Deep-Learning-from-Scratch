from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor_initialShape = input_tensor.shape   
        output = np.array([arr.flatten() for arr in input_tensor]) 
        return output
        
    def backward(self, error_tensor):
        output = error_tensor.reshape(self.input_tensor_initialShape)
        return output