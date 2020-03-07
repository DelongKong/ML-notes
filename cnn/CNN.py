# -*- coding: utf-8 -*-

import numpy as np

def get_receptive_field(input_data, height_index, width_index, height, \
                        width, stride):
    height_start_index = height_index * stride
    width_start_index = width_index * stride
        
    return input_data[height_start_index : height_start_index+height, \
                      width_start_index : width_start_index+width]

def padding(input_data, padding_num):
    if padding_num == 0:
        return input_data

def convolution(input_data, kernel, output_data, stride, bias):
    output_height = output_data.shape[0]
    output_width = output_data.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    
    for i in range(output_height):
        for j in range(output_width):
            receptive_field = get_receptive_field(input_data, i, j, \
                                                  kernel_height, \
                                                  kernel_width, stride)
            kernel_value = np.sum(np.multiply(receptive_field, kernel))
            output_data[i][j] = kernel_value + bias

# op is a function. Read element in array, op(it), write in array again
def element_wise_op(array, op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)

def get_max_index(data_array):
    position = np.where(data_array == np.max(data_array))
    return (position[0][0], position[1][0])

def max_pool(input_data, sambling):
    temp = input_data 
    
    return max(temp.flatten()), get_max_index(temp)

class Kernel(object):
    
    def __init__(self, height, width, num):
        self.weights = Kernel.weights_init(height, width, num)
        self.bias = Kernel.bias_init()
        self.weights_grad = np.zeros((height, width))
        self.bias_grad = 0
        
    @staticmethod
    def weights_init(height, width, num):
        # MSRA initialization
        return np.random.normal(0, (2/(height*width*num))**0.5, \
                                (height, width))
    @staticmethod
    def bias_init():
        return 0
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

class ConvolutionalLayer(object):
    
    def __init__(self, input_size, kernel_size, kernel_num, padding, stride, \
                 activator, learning_rate):
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.kernel_num = kernel_num
        self.padding = padding
        self.stride = stride
        self.activator = activator
        self.l_rate = learning_rate
        
        self.output_height = int((self.input_height - self.kernel_height + \
                              2 * self.padding) / self.stride + 1)
        self.output_width = int((self.input_width - self.kernel_width + \
                              2 * self.padding) / self.stride + 1)                     
        self.output_datas = np.zeros((self.kernel_num, self.output_height, \
                                self.output_width))        
        self.kernels = []
        for i in range(self.kernel_num):
            self.kernels.append(Kernel(self.kernel_height, self.kernel_width, \
                                       self.kernel_num))
        self.activator = activator
        self.l_rate = learning_rate
      
    def forward(self, input_data):
        self.input_data = input_data
        self.padded_data = padding(input_data, 0)
        for j in range(len(self.kernels)):
            convolution(self.padded_data, self.kernels[j].weights, \
                                      self.output_datas[j], self.stride, \
                                      self.kernels[j].get_bias())
            
        
        element_wise_op(self.output_datas, self.activator.forward)
        
    def backward(self):
        pass
    
    def update(self, delta_, firstdata):
        for n in range(self.kernel_num):
            convolution(firstdata, delta_[n], self.kernels[n].weights_grad, self.stride, 0)
            self.kernels[n].bias_grad = np.sum(delta_[n])
            self.kernels[n].update(self.l_rate)
    

class MaxPoolingLayer(object):
    
    def __init__(self, input_size, sambling_size, sambling_num, stride):
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.sambling = np.eye(sambling_size[0], sambling_size[1])
        self.sambling_height = sambling_size[0]
        self.sambling_width = sambling_size[1]
        self.sambling_num = sambling_num
        self.stride = stride
        self.max_position = []
        self.delta = np.zeros((self.sambling_num, self.input_height, \
                               self.input_width))
        
        self.output_height = int((self.input_height - \
                              self.sambling_height) / stride + 1)
        self.output_width = int((self.input_width - \
                             self.sambling_width) / stride + 1)
        self.output_datas = np.zeros((self.sambling_num, self.output_height, \
                                      self.output_width))
        
    
    def forward(self, input_data):
        for n in range(self.sambling_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    '''
                    test = get_receptive_field(input_data[n], \
                                                         i, j, \
                                                         self.sambling_height, \
                                                         self.sambling_width, \
                                                         self.stride)
                    '''
                    #print(test.shape)
                    
                    self.output_datas[n][i][j], n_posi = max_pool(\
                                     get_receptive_field(input_data[n], \
                                                         i, j, \
                                                         self.sambling_height, \
                                                         self.sambling_width, \
                                                         self.stride), \
                                                         self.sambling)
                    self.max_position.append(n_posi)
    
    def backward(self, delta_):
        for n in range(self.sambling_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    k, l = self.max_position[n][0], self.max_position[n][1]
                    new_i, new_j = i*self.stride+k, j*self.stride+l
                    self.delta[n][new_i][new_j] = delta_[n][i][j]
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    