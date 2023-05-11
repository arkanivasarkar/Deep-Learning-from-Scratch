import numpy as np
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from copy import deepcopy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.trainable = True
        self._memorize = False
        self._optimizer = None
        self.regularizer = None
 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_xh =FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_hy = FullyConnected(self.hidden_size, self.output_size)
        self._gradient_weights = np.zeros((self.fc_xh.weights.shape))
        self.weights = self.fc_xh.weights

        self.gradient_weights_new = np.zeros((self.fc_xh.weights.shape))

        self.hidden_values = np.zeros(self.hidden_size)
        self.opt1 = None
        self.opt2 = None
        self._optimizer = None
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self.ht_unactiv = []
        self.y_unactiv = []

    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.opt1 = deepcopy(opt)
        self.opt2 = deepcopy(opt)
        self._optimizer = self.opt1
      
    
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize


    def forward(self, input_tensor):
        self.input_value = input_tensor
        batch_dim = input_tensor.shape[0]
        self.ht = np.zeros((batch_dim, self.hidden_size))
        self.output = np.empty((batch_dim, self.output_size))

        if self.memorize == False:
            self.hidden_values = np.zeros(self.hidden_size)

        for t in range(batch_dim):
            x = input_tensor[t, ...]
            x_conc = np.concatenate((x, self.hidden_values), axis= None).reshape(1,-1)            
            ht_unactivated = self.fc_xh.forward(x_conc)
            self.hidden_values = self.tanh.forward(ht_unactivated)
            y_unactivated = self.fc_hy.forward(self.hidden_values)
            y = self.sigmoid.forward(y_unactivated)
            self.ht[t] = self.hidden_values
            self.output[t] = y

        return self.output
    


    def backward(self, error_tensor):
        output_err = np.zeros(self.input_value.shape)
        prev_h = 0

        gweights1 = np.zeros((self.fc_xh.weights.shape))
        gweights2 = np.zeros((self.fc_hy.weights.shape))

        
        for t in range(error_tensor.shape[0]-1, -1, -1):
            err = error_tensor[t, ...]
            Sigmoid.act_output = self.output[t]
            err = self.sigmoid.backward(err)            
        
            err = self.fc_hy.backward(err.reshape(1,-1))
            gweights2 = gweights2 + self.fc_hy.gradient_weights

            err = err + prev_h
            TanH.act_output = self.ht[t]
            err = self.tanh.backward(err)

            if t == 0:
                hidd = np.zeros((1, self.hidden_size))
            else:
                hidd = self.ht[t]

            x_conc = np.concatenate((self.input_value[t], hidd, [1]), axis=None)
            self.fc_xh.input_tensor = x_conc.reshape(1,-1)

            err = self.fc_xh.backward(err)
            output_err[t] = err[:, :self.input_size]
            prev_h = err[:, self.input_size:]

            gweights1 =  gweights1 + self.fc_xh.gradient_weights
           
        if self._optimizer != None:
            self.fc_hy.weights = self._optimizer.calculate_update(self.fc_hy.weights, gweights2)
            self.fc_xh.weights = self._optimizer.calculate_update(self.fc_xh.weights, gweights1)

        return output_err

    @property
    def weights(self):
        return self.fc_xh.weights

    @weights.setter
    def weights(self, weights):
        self.fc_xh.weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_new

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fc_hy._gradient_weights = gradient_weights
        self.fc_hx._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_xh.initialize(weights_initializer, bias_initializer)
        self.fc_hy.initialize(weights_initializer, bias_initializer)