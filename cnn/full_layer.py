# -*- coding: utf-8 -*-

import math
import numpy as np
from Helpers import helper as hp


class FullConnectedLayer(object):
    
    def __init__(self, dim_input, dim_output, activator, learning_rate, \
                 output_layer=False):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activator = activator
        self.l_rate = learning_rate
        self.W = FullConnectedLayer.weight_init(dim_input, dim_output)
        self.b = FullConnectedLayer.bias_init(dim_output)
        self.output_layer = output_layer
        self.out_data = np.zeros((dim_output, 1))
        
    @staticmethod   
    def weight_init(dim_input, dim_output):
        # Xavier initialization
        sigema = math.sqrt(1/dim_input)
        return np.random.normal(0, sigema, (dim_output, dim_input))
    @staticmethod
    def bias_init(dim_output):
        # 0 initialization
        return np.zeros((dim_output, 1))
        
    def forward(self, input_data):
        self.input = input_data
        self.Z = hp.mdot(self.W, input_data) + self.b
        # print(self.Z.shape)
        if self.output_layer == False:
            self.out_data = self.activator.forward(self.Z)
        else:
            self.out_data = self.Z
      
    def backward(self, delta_input, W_input):
        if self.output_layer == False:
            # activa_grad = self.activator.backward(self.Z)
            # print()
            self.delta = np.multiply(hp.mdot(W_input.T, delta_input), 1)
        else:
            self.delta = delta_input
        
        self.W_grad = hp.mdot(self.delta, self.input.T)
        self.b_grad = self.delta
         
    def updata_para(self):
        self.W -= self.l_rate * self.W_grad
        self.b -= self.l_rate * self.b_grad
        
        