from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.trainable = False
        self.stride_shape= stride_shape
        self.pooling_shape= pooling_shape
       

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        input_mat_shape = input_tensor.shape[2:]
        pool_shape = np.array(self.pooling_shape)
        stride_shape = np.array(self.stride_shape)
        
        maxPooled_shape = np.floor_divide(np.subtract(input_mat_shape, pool_shape), stride_shape, dtype=int) + 1 #matrix size after pooling calculation        
        final_shape = [input_tensor.shape[0],input_tensor.shape[1],maxPooled_shape[0],maxPooled_shape[1]] # with batches and channels
       
        output = np.zeros((final_shape), dtype=float)
        self.maximaLoc = []

        for batchIndx in range(final_shape[0]):
            for channelIndx in range(final_shape[1]):
                for imRow in range(final_shape[2]):
                    for imCol in range(final_shape[3]):      
                        #max value in pool
                        output[batchIndx,channelIndx,imRow,imCol] = np.max(input_tensor[batchIndx, channelIndx,
                        imRow * self.stride_shape[0]:imRow  * self.stride_shape[0] + self.pooling_shape[0], 
                        imCol * self.stride_shape[1]:imCol * self.stride_shape[1] + self.pooling_shape[1]])

                        #index of max value
                        maxVal_index = np.argwhere(input_tensor[batchIndx, channelIndx, 
                        imRow * self.stride_shape[0]:imRow  * self.stride_shape[0] + self.pooling_shape[0], 
                        imCol * self.stride_shape[1]:imCol * self.stride_shape[1] + self.pooling_shape[1]] == output[batchIndx,channelIndx,imRow,imCol]) + [imRow * self.stride_shape[0], imCol * self.stride_shape[1]]
                        maxVal_index = np.array([batchIndx, channelIndx, maxVal_index[0][0], maxVal_index[0][1]])
  
                        self.maximaLoc.append(maxVal_index)                        
        return output




    def backward(self, error_tensor):
        error_shape = error_tensor.shape

        output = np.zeros((self.input_tensor_shape),dtype=float)
        output_shape = output.shape
        
        count = -1
        for batchIndx in range(output_shape[0]):
            for channelIndx in range(output_shape[1]):
                for imRow in range(0,error_shape[2]):
                    for imCol in range(0,error_shape[3]): 
                        count += 1
                        #error values added to maxima locations
                        output[batchIndx, channelIndx, self.maximaLoc[count][2], self.maximaLoc[count][3]] += error_tensor[batchIndx, channelIndx, imRow, imCol]
        return output
        





