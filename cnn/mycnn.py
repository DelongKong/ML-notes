# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist

import CNN
import full_layer
import activators

def norm(label):
    label_vec = []
    label_value = label  
    for i in range(10):
        if i == label_value:
            label_vec.append(1)
        else:
            label_vec.append(0)
    return np.array(label_vec).reshape(-1, 1)

class MyCnn(object):
    
    def __init__(self):
        self.C1 = CNN.ConvolutionalLayer((28,28), (5,5), 6, 0, 1, \
                                         activators.ReluActivator(), 0.001)
        self.S1 = CNN.MaxPoolingLayer((24,24), (2,2), 6, 2)
        self.OutLayer = full_layer.FullConnectedLayer(864, 10, \
                                                 activators.SigmoidActivator(), \
                                                 0.001)
        
    def forward(self, first_data):
        #print(first_data.shape)
        self.C1.forward(first_data)
        #print(self.C1.output_datas.shape)
        self.S1.forward(self.C1.output_datas)
        s1_out = self.S1.output_datas.flatten().reshape(-1, 1)
        self.OutLayer.forward(s1_out)
        
        return self.OutLayer.out_data
        
    def backward(self, first_data, labels):
        delta_L = self.OutLayer.out_data - labels
        # print(self.OutLayer.W.shape)
        self.OutLayer.backward(delta_L, self.OutLayer.W)
        undo_delta = self.OutLayer.delta.reshape(6, 12, 12)
        self.S1.backward(undo_delta)
        
        self.C1.update(self.S1.delta, first_data)
        
        
if __name__ == '__main__':
    myCNN = MyCnn()
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    A_ = X_train[0:100,:,:]
    Labels = y_train[0:100]
    norm_labels = []
    for i in range(100):
        norm_labels.append(norm(Labels[i]))
    for j in range(10):
        print("iterate:", j)
        for n in range(100):
            first_data = A_[n]
            result = myCNN.forward(first_data)
            myCNN.backward(first_data, norm_labels[n])
    print("next test")
    
    first_data = X_test[1]
    result = myCNN.forward(first_data)
    print(result)
    print(y_test[1])     
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        