from copy import deepcopy
import numpy as np
np.random.seed(0)

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.loss = []
        self.layers = []
        self.optimizer = optimizer
        self.data_layer = []
        self.loss_layer = []
        self.label_tensor = []
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    
    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        regularization_loss = 0
        for layer in self.layers:
            layer.testing_phase = False
            input_tensor = layer.forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, self.label_tensor)

        regularization_loss = 0
        if self.optimizer.regularizer:
            regularization_loss = self.optimizer.regularizer.norm(loss)
        return loss + regularization_loss
    


    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for iter in range(iterations):
            #print('Iteration ',iter,' of', iterations)
            l = self.forward()
            self.backward()
            self.loss.append(l)

    
    def test(self, input_tensor):     
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor



