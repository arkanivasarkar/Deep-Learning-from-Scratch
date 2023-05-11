from Layers.Base import BaseLayer
import numpy as np 
np.random.seed(0)

class Constant(BaseLayer):
    def __init__(self, weight_init=0.1):
        super().__init__()
        self.weight_init = weight_init

    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.tile(self.weight_init, weights_shape)
        return output



class UniformRandom(BaseLayer):
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        output = np.random.uniform(0, 1, weights_shape)
        return output



class Xavier(BaseLayer):
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = (2/(fan_out + fan_in))**0.5
        output = np.random.normal(0, sigma, weights_shape)
        return output



class He(BaseLayer):
    def __init__(self):
        super().__init__()

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = (2/fan_in)**0.5
        output = np.random.normal(0, sigma, weights_shape)
        return output
