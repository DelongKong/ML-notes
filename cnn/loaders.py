# -*- coding: utf-8 -*-

import numpy as np

class Loader(object):
    
    def __init__(self, path, samples):
        self.path = path
        self.samples = samples
        
    def get_raw_data(self):
        f = open(self.path, 'rb')
        raw_data = f.read()
        f.close()
        return raw_data
        
class ImageLoader(Loader):
    def get_image(self, raw_data, index):
        # images = [] # list: store all 28x28 image data
        start = 16 + index*28*28
        end = start + 28*28
        image = np.array(raw_data[start:end]).reshape(28, 28)
        
        return image
    


if __name__ == "__main__":
    image = ImageLoader("data/train-images.idx3-ubyte", 1)
    print(type(image.get_raw_data))
        