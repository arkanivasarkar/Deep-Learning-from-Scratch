from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import numpy as np
import sys


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()

        self._gradient_bias = []
        self._gradient_weights = []
        self._optimizer = [] 
        self._bias_optimizer = []

        self.mean = 0
        self.variance = 1
        self.mean_test = 0
        self.variance_test = 1



    def initialize(self, weights_initializer=[], bias_initializer=[]):
        self.bias = np.zeros((self.channels),dtype=float) # beta
        self.weights = np.ones((self.channels),dtype=float) # gamma


    def forward(self, input_tensor, decay = 0.8):
        self.input_tensor = input_tensor

        if len(input_tensor.shape) == 2:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            if self.testing_phase:
                self.x_cap = (input_tensor - self.mean_test)/(self.variance_test + sys.float_info.epsilon)**0.5

            else:
                current_mean = np.mean(input_tensor, axis=0)
                current_variance = np.var((input_tensor), axis=0)

                self.mean_test = self.mean*decay + (1 - decay) * current_mean
                self.variance_test = self.variance*decay + (1 - decay) * current_variance

                self.mean = current_mean
                self.variance = current_variance
                self.x_cap = (input_tensor - self.mean)/(self.variance + sys.float_info.epsilon)**0.5
            output = self.weights*self.x_cap + self.bias


        elif len(input_tensor.shape) == 4:
            self.mean = np.mean(input_tensor, axis=(0, 2, 3))
            self.variance = np.var((input_tensor), axis=(0, 2, 3))
            
            if self.testing_phase:
                self.x_cap = (input_tensor - self.mean_test.reshape((1, input_tensor.shape[1], 1, 1)))/(self.variance_test.reshape((1, input_tensor.shape[1], 1, 1)) + sys.float_info.epsilon)**0.5

            else:
                current_mean = np.mean(input_tensor, axis=(0, 2, 3))
                current_variance = np.var(input_tensor, axis=(0, 2, 3))

                self.mean_test = decay*self.mean.reshape((1, input_tensor.shape[1], 1, 1)) + (1 - decay) * current_mean.reshape((1, input_tensor.shape[1], 1, 1))
                self.variance_test = decay*self.variance.reshape((1, input_tensor.shape[1], 1, 1)) + (1 - decay) * current_variance.reshape((1, input_tensor.shape[1], 1, 1))

                self.mean = current_mean
                self.variance = current_variance

                self.x_cap = (input_tensor - self.mean.reshape((1, input_tensor.shape[1], 1, 1)))/(self.variance.reshape((1, input_tensor.shape[1], 1, 1)) + sys.float_info.epsilon)**0.5
            output = self.weights.reshape((1, input_tensor.shape[1], 1, 1))*self.x_cap + self.bias.reshape((1, input_tensor.shape[1], 1, 1))

        return output


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            output = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor),self.weights, self.mean, self.variance)
            output = self.reformat(output)
            self.gradient_weights = np.sum(error_tensor*self.x_cap, axis=(0,2,3))
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))

        elif len(error_tensor.shape) == 2:
            output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance)
            self.gradient_weights = np.sum(error_tensor*self.x_cap, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return output


    def reformat(self, tensor):
        #im2vec tensor
        if len(tensor.shape) == 4:
            tensor_shape = tensor.shape
            tensor = tensor.reshape((tensor_shape[0], tensor_shape[1], tensor_shape[2]*tensor_shape[3]))
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(tensor_shape[0] * tensor_shape[2] * tensor_shape[3], tensor_shape[1])   

        #vec2im tensor
        else:
            tensor = tensor.reshape((self.input_tensor.shape[0], self.input_tensor.shape[2]*self.input_tensor.shape[3], self.input_tensor.shape[1]))
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(self.input_tensor.shape[0], self.input_tensor.shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3])

        return tensor



    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self._gradient_weights = val


    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, val):
        self._gradient_bias = val


    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val


    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, val):
        self._bias_optimizer = val
 