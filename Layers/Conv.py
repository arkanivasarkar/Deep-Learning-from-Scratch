import numpy as np
from Layers.Base import BaseLayer
from scipy import signal
from copy import deepcopy
np.random.seed(0)

class Conv(BaseLayer):
    def __init__(self, stride_shape = np.random.uniform(0,1), 
     convolution_shape = np.random.uniform(0,2), num_kernels = np.random.uniform(0,1)):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0,1,(self.num_kernels, *self.convolution_shape)) 
        self.bias = np.random.uniform(0,1,(self.num_kernels)) 

        self.gradient_weights = []
        self.gradient_bias = []
        self._optimizer = []
    

 
    def forward(self, input_tensor):
        self.prev_input = input_tensor
        channels = input_tensor.shape[1]
        batch_size = input_tensor.shape[0]

        output = np.zeros((batch_size, self.num_kernels, *input_tensor.shape[2:]))        
        for batchIndx in range(batch_size):
            for channelIndx in range(self.num_kernels):
                for ii in range(channels):
                    output[batchIndx, channelIndx] += signal.correlate(input_tensor[batchIndx, ii], self.weights[channelIndx,ii], "same")
                
                output[batchIndx, channelIndx] += self.bias[channelIndx]        
        output = self.apply_stride(output)
        return output
    
    
    def apply_stride(self, output):
        if len(self.convolution_shape)==3:
            return output[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]]
        else:
            return output[:, :, 0::self.stride_shape[0]]


    def upsample_stride(self, input_tensor, out_shape):        
        result = np.zeros(out_shape)
        if len(out_shape) == 2: 
            result[0::self.stride_shape[0], 0::self.stride_shape[1]] = input_tensor
            return result
        else:
            result[0::self.stride_shape[0]] = input_tensor
            return result


    def pad(self, input_tensor):
        filters = np.array(self.convolution_shape[1:])
        pad_ini = np.floor(filters/2).astype(int)       
        pad_final = filters - pad_ini - 1  # for uneven
        if len(filters) == 2: # 2D
            pad_width = [(0,0), (0,0), (pad_ini[0], pad_final[0]), (pad_ini[1],pad_final[1])]
        else:
            pad_width = [(0,0), (0,0), (pad_ini[0], pad_final[0])]

        padded_input = np.pad(input_tensor, pad_width=pad_width, constant_values=0)
        return padded_input



    def backward(self, error_tensor):       
        self.gradient_weights = np.zeros(self.weights.shape)
        if (len(self.convolution_shape)==3):
            axis = (0,2,3)
        else:
            axis = (0,2)
        self.gradient_bias = np.sum(error_tensor, axis=axis)
        new_error_tensor = np.zeros(self.prev_input.shape)
        channels = self.prev_input.shape[1]
        padded_input = self.pad(self.prev_input)

        for batchIndx in range(self.prev_input.shape[0]):
            for kernelNum in range(self.num_kernels):
                for ii in range(channels):
                    upsampled_error = self.upsample_stride(error_tensor[batchIndx, kernelNum], self.prev_input.shape[2:])
                    out_w = signal.correlate(padded_input[batchIndx, ii], upsampled_error, "valid")
                    self.gradient_weights[kernelNum, ii] += out_w

                    out_err = signal.convolve(upsampled_error, self.weights[kernelNum, ii], "same")
                    new_error_tensor[batchIndx, ii] += out_err

        if  self._optimizer != []:
            self.weights = self._optimizer[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self.gradient_bias)
        return new_error_tensor



    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        weight_opt = deepcopy(value)
        bias_opt = deepcopy(value)
        self._optimizer = [weight_opt, bias_opt]

