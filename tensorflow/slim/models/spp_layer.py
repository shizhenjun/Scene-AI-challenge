from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math
def spatial_pyramid_pooling(inputdata, out_pool_size):
    '''
    spatial_pyramid_pooling and flatten it(generally followed by fc layer)
    input:a tensor from previous layer
    out_pool_size:a int vertor of expected output size of max pooling;
    out_pool_size = [3, 2, 1]
    '''
    batch_size = inputdata.get_shape().as_list()[0]
    pre_h = float(inputdata.get_shape().as_list()[1])
    pre_w = float(inputdata.get_shape().as_list()[2])

    for i in range(len(out_pool_size)):
        h_strd = int(math.floor(pre_h/float(out_pool_size[i])))
        w_strd = int(math.floor(pre_w/float(out_pool_size[i])))
        h_wind = int(math.ceil(pre_h/float(out_pool_size[i])))
        w_wind = int(math.ceil(pre_w/float(out_pool_size[i])))

        max_pool = tf.nn.max_pool(inputdata,
                                  ksize=[1, h_wind, w_wind, 1],
                                  strides=[1, h_strd, w_strd, 1],
                                  padding='VALID')
        if(i == 0):
            sppout = tf.reshape(max_pool, [batch_size, -1])
        else:
            sppout = tf.concat(values=[sppout, tf.reshape(max_pool, [batch_size, -1])], axis=1)

    sppnum = sppout.get_shape().as_list()[1]
    return sppout, sppnum
