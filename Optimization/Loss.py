import numpy as np
import sys

class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = []

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor        
        input_tensor = input_tensor[np.nonzero(label_tensor)]
        input_tensor = input_tensor + sys.float_info.epsilon 
        output = -np.sum(np.log(input_tensor))
        return output

    def backward(self, label_tensor):
        output = -(label_tensor / self.input_tensor)
        return output



