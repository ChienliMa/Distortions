# -*- coding:utf-8 -*-
import distortions
from scipy import misc
import numpy as np
from matplotlib import pyplot as plt, cm

def run_example( size = 28 ):
    """
    Run this file and the result will be saved as result.jpg
    Buttle neck:
    local_elastic
    pintch
    uniform_noist
    salt_and_pepper_noise
    """
    methods = """
            affine
            local_elastic                  slant
            pinch                          thickness                            
            contrast                       salt_and_pepper_noise
            gaussian_noise                 uniform_noise 
            scratch                        permute
            smooth                         blur
            """
    img = misc.imresize( misc.imread('./imgs/H.jpg'), (size,size,3) )
    methods = methods.split()
    num_of_methods = len( methods )
    result = np.zeros( [ size*2, size*(num_of_methods//2+2), 3 ])
    result[:size:,:size:,:] = img
    for index in xrange( num_of_methods ):
        y = index // 2 + 1
        x = index % 2
        tmp= distortions.distort( img, methods[index] )
        result[ x*size:(x+1)*size, y*size:(y+1)*size, : ] = tmp 
    result = result.mean(axis=2)
    misc.imsave('result.jpg',result)

if __name__ == '__main__':
    run_example()
