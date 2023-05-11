from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor) 
        self.output = np.exp(input_tensor)/np.expand_dims(np.sum(np.exp(input_tensor), axis=1), axis=1)
        return self.output      


    def backward(self, error_tensor):
        loss_grad = error_tensor - np.expand_dims(np.sum(error_tensor * self.output, axis=1), axis=1)
        output = self.output * loss_grad
        return output

