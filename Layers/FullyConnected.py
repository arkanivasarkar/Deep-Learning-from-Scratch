from Layers.Base import BaseLayer
import numpy as np 
np.random.seed(0)

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0,1,(input_size + 1, output_size)) 
        self._optimizer = []
        self._gradient_weights = np.zeros((self.weights.shape),dtype=float)


    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        bias = np.expand_dims(np.ones(batch_size),axis=1) #column matrix
        self.input_tensor = np.hstack((input_tensor, bias))
        output = np.dot(self.input_tensor, self.weights)
        return output


    def backward(self, error_tensor):
        loss_grad = np.dot(error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer: 
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        output = loss_grad[:,0:loss_grad.shape[1]-1] #remove column of bias
        return output


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, self.bias))

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer


    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizerType):
        self._optimizer = optimizerType
    @optimizer.deleter
    def optimizer(self):
        del self._optimizer


 
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, grad_weight):
        self._gradient_weights = grad_weight
    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

