from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math
def fixed_size_pooling(inputdata, pool_h, pool_w):
    '''
    function:pool feature map into fixed_size
    inputdata:a tensor from previous layer
    pool_h: height
    pool_w: width
    '''
    pre_h = float(inputdata.get_shape().as_list()[1])
    pre_w = float(inputdata.get_shape().as_list()[2])
    
    h_strd = int(math.floor(pre_h/float(pool_h)))
    w_strd = int(math.floor(pre_w/float(pool_w)))
    h_wind = int(math.ceil(pre_h/float(pool_h)))
    w_wind = int(math.ceil(pre_w/float(pool_w)))

    max_pool = tf.nn.max_pool(inputdata, ksize=[1, h_wind, w_wind, 1], strides=[1, h_strd, w_strd, 1], padding='VALID')

    return max_pool