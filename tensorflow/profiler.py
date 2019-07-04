

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
                         Convolution2D, MaxPooling2D, BatchNormalization 
                                    
import time
from math import log
#tf.enable_eager_execution()

LAYERTYPES = {
    'CNN': Convolution2D,
    'Pooling': MaxPooling2D,
    'Dense': Dense,
    'Relu': Activation,
    'BatchNormalization': BatchNormalization,
    'Dropout': Dropout
}

class Profiler():
  def __init__(self, layer_type, num_inputs=6, num_layers=10):
    self.regressor = None
    self.layer_type = layer_type
    self.num_inputs = num_inputs
    self.num_layers = num_layers
    self.profiling()

  def input_sampler(self):
    #widths = [227, 55, 27, 13, 7, 5, 27, 13, 7, 5]
    widths = [150, 55, 27, 13, 7, 5, 27, 13, 7, 5]
    #channels = [3, 96, 256, 384, 552, 1024, 3, 96, 256, 384]
    channels = [3, 12, 24, 36, 48, 60, 220, 330, 36, 420]
    return zip(widths, widths, channels)


  def layer_sampler(self):
    if self.layer_type=='CNN':
      kernels = [11, 11, 9, 9, 7, 7, 5, 5, 3, 3]
      strides = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
      #num_filters = [96, 256, 384, 256, 384, 256, 384, 256, 384, 256]
      num_filters = [56, 106, 156, 106, 156, 106, 156, 106, 156, 106]
      return zip(num_filters, kernels, strides)

    elif self.layer_type=='Dense':
      #output_dim = [4, 8, 16, 32, 64, 128, 256, 412, 1024, 4096]
      output_dim = [4, 8, 16, 32, 44, 54, 64, 74, 84, 104]
      return [[i] for i in output_dim]

    elif self.layer_type=='Pooling':
      pool_size = [3, 5, 7, 9, 11, 13, 15]
      return [(i, i) for i in pool_size]
    elif self.layer_type=='Dropout':
      return [[0.5]]
    elif self.layer_type=='Relu':
      return [['relu']]
    else:
      return [[]]

  def profiling(self):
    input_data_size = self.input_sampler()
    layer_specs = self.layer_sampler()

    x1s = []
    x2s = []
    ys = []
    bs = 10
    for width, height, channel in input_data_size:
      inputs = tf.random_uniform([bs, width, height, channel])
      #inputs = tf.contrib.eager.Variable(inputs, name='weights') 
      print(inputs.shape)
      for layer_spec in layer_specs:
        # X1： number of features in the input feature maps
        x1s.append(width * height * channel)
        # X2: computation per pixel
        if self.layer_type=='CNN':
          x2s.append((layer_spec[0] / layer_spec[1]) ** 2 * layer_spec[1])
        elif self.layer_type=='Dense':
          x2s.append(layer_spec[0])
        elif self.layer_type=='Pooling':
          x2s.append((width - layer_spec[0])**2 * channel)

        model = LAYERTYPES[self.layer_type](*layer_spec)
        start = time.time()
        outputs = model(inputs)
        ys.append(time.time() - start)

    if not x2s:
      X = tf.transpose(tf.convert_to_tensor([x1s, [1]*len(x1s)], dtype=tf.float32))
    else:
      X = tf.transpose(tf.convert_to_tensor([x1s, x2s, [1]*len(x1s)], dtype=tf.float32))
    y = tf.transpose(tf.convert_to_tensor([ys], dtype=tf.float32))


    self.regressor = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), \
                     tf.transpose(X)), y)

  def predict(self, features):
    x = tf.convert_to_tensor([features + [1]], dtype=tf.float32)
    return tf.matmul(x, self.regressor).numpy()[0][0]
    # return tf.random_uniform([1])


#profilers = {}
#for t in LAYERTYPES.keys():
  #print(t)
  #profilers[t] = Profiler(t)

#for t in LAYERTYPES.keys():
  #print(t, profilers[t].regressor)
#mobile_layers = [('Dense', [10, 20]), ('CNN', [120, 50])]


#ES_mobile = [profilers[ltype].predict(size) for ltype, size in mobile_layers]
