from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.probability = probability
        self.trainable = False


    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            randDropMat = np.random.uniform(0,1,(input_tensor.shape[0], input_tensor.shape[1]))
            self.randDropMask = randDropMat < self.probability
            self.randDropMask =  self.randDropMask*(1/self.probability)
            output = input_tensor * self.randDropMask
        return output



    def backward(self, error_tensor):
        output = error_tensor*self.randDropMask
        return output