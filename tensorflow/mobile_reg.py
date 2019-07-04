import tensorflow as tf
import os
import tensorflow.contrib.eager as tfe
import numpy as np
import time
import matplotlib.pyplot as plt
import socket
import pickle
import json

from tensorflow.python.training import checkpointable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
                                    Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import ReLU                                    

#eager execution enabled
tf.enable_eager_execution()

#importing profiler
from profiler import Profiler


#Model implemented

class MyModel(tf.contrib.eager.Checkpointable):
 def AlexNet():
    model = tf.keras.Sequential()
    model.add(Convolution2D(64, (11, 11), padding='valid', name='conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    

    model.add(Convolution2D(128, (7, 7), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    

    model.add(Convolution2D(192, (3, 3), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))


    model.add(Convolution2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    
    
    model.add(Convolution2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    

    #model.add(Flatten())

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
   
    

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model



#number of partitions
num_of_partitions =30



#functions for partition call
def mobile_call(model, inputs, layer_end):
  for layer in model.layers[:layer_end]:
    inputs = layer(inputs)
  return inputs


def server_call(model, inputs, layer_start):
  for layer in model.layers[layer_start:]:
    inputs = layer(inputs)
  return inputs


#execution time calculation
def execution_time():
  rt = np.zeros(num_of_partitions)
  #for bwn in range(bw_data):
  for inr in range(num_of_partitions):
      rt[inr]=(serv_time[inr]+mob_time[inr])
      #rt[inr]=(serv_time[inr]+mob_time[inr]+input_rate[0])
  return rt



#model building 

model = MyModel.AlexNet()


mob_time=np.zeros(num_of_partitions)


#Server_profiler1 = Profiler('CNN')
#Server_profiler2 = Profiler('CNN')
#Server_profiler3 = Profiler('CNN')
#Server_profiler4 = Profiler('Dense')  

mobile_profiler1 = Profiler('CNN')
mobile_profiler2 = Profiler('BatchNormalization')
mobile_profiler4 = Profiler('Pooling')
mobile_profiler3 = Profiler('Relu')
#mobile_profiler5 = Profiler('Dense')
#mobile_profiler6 = Profiler('Dropout')
mobile_profiler7 = Profiler('Dense')

#serv_time1 = Server_profiler1.predict([120,50])
#serv_time[0]=serv_time1
#print(serv_time1)

conv_in1=120 
conv_out1=50
pool_in1=120
pool_out1=50
conv_in2=120 
conv_out2=50
conv_in3=120 
conv_out3=50
conv_in4=120 
conv_out4=50
dense_in1=10
conv_in5=120 
conv_out5=50
dense_out1=20
dense_in2=10
dense_out2=20
dense_in3=10
dense_out3=20

print(mob_time[0])

mob_time1 = mobile_profiler1.predict([conv_in1, conv_out1])
mob_time[1]=mob_time1
print(mob_time[1])

#serv_time2 = Server_profiler1.predict([120,50])+Server_profiler2.predict([120,50])
#serv_time[1]=serv_time2
#print(serv_time2)

mob_time2 = mob_time1+mobile_profiler2.predict([conv_out2])
mob_time[2]=mob_time2
print(mob_time[2])

mob_time3 = mob_time2+mobile_profiler3.predict([conv_out2])
mob_time[3]=mob_time3
print(mob_time[3])


mob_time4 = mob_time3+mobile_profiler4.predict([pool_in1,pool_out1])
mob_time[4]=mob_time4
print(mob_time[4])

#serv_time3 = Server_profiler1.predict([120,50])+Server_profiler2.predict([120,50])+Server_profiler3.predict([120,50])
#serv_time[2]=serv_time3
#print(serv_time3)

mob_time5 = mob_time4 +mobile_profiler1.predict([conv_in2, conv_out2])

mob_time[5]=mob_time5
print(mob_time[5])

mob_time6 = mob_time5+mobile_profiler2.predict([conv_out1])
mob_time[6]=mob_time6
print(mob_time[6])

mob_time7 = mob_time6+mobile_profiler3.predict([conv_out1])
mob_time[7]=mob_time7
print(mob_time[7])


mob_time8 = mob_time7+mobile_profiler4.predict([pool_in1,pool_out1])
mob_time[8]=mob_time8
print(mob_time[8])

mob_time9 = mob_time8+mobile_profiler1.predict([conv_in3, conv_out3])
mob_time[9]=mob_time9
print(mob_time[9])

mob_time10 = mob_time9+mobile_profiler2.predict([conv_out1])
mob_time[10]=mob_time10
print(mob_time[10])

mob_time11 = mob_time10+mobile_profiler3.predict([conv_out1])
mob_time[11]=mob_time11
print(mob_time[11])

mob_time12 = mob_time11+mobile_profiler4.predict([pool_in1,pool_out1])
mob_time[12]=mob_time12
print(mob_time[12])

mob_time13 = mob_time12+mobile_profiler1.predict([conv_in4, conv_out4])
mob_time[13]=mob_time13
print(mob_time[13])

mob_time14 = mob_time13+mobile_profiler2.predict([conv_out1])
mob_time[14]=mob_time14
print(mob_time[14])

mob_time15 = mob_time14+mobile_profiler3.predict([conv_out1])
mob_time[15]=mob_time15
print(mob_time[15])

mob_time16 = mob_time15+mobile_profiler4.predict([pool_in1,pool_out1])
mob_time[16]=mob_time16
print(mob_time[16])

mob_time17 = mob_time16+mobile_profiler1.predict([conv_in5, conv_out5])
mob_time[17]=mob_time17
print(mob_time[17])

mob_time18 = mob_time17+mobile_profiler2.predict([conv_out1])
mob_time[18]=mob_time18
print(mob_time[18])

mob_time19 = mob_time18+mobile_profiler3.predict([conv_out1])
mob_time[19]=mob_time19
print(mob_time[19])

mob_time20 = mob_time19+mobile_profiler4.predict([pool_in1,pool_out1])
mob_time[20]=mob_time20
print(mob_time[20])


mob_time21 = mob_time20+mobile_profiler7.predict([dense_in1, dense_out1])
mob_time[21]=mob_time21
print(mob_time[21])

mob_time22 = mob_time21+mobile_profiler2.predict([conv_out1])
mob_time[22]=mob_time22
print(mob_time[22])

mob_time23 = mob_time22+mobile_profiler3.predict([conv_out1])
mob_time[23]=mob_time23
print(mob_time[23])


mob_time24 = mob_time23+mobile_profiler7.predict([dense_in2, dense_out2])
mob_time[24]=mob_time24
print(mob_time[24])

mob_time25 = mob_time24+mobile_profiler2.predict([conv_out1])
mob_time[25]=mob_time25
print(mob_time[25])

mob_time26 = mob_time25+mobile_profiler3.predict([conv_out1])
mob_time[26]=mob_time26
print(mob_time[26])

mob_time27 = mob_time26+mobile_profiler7.predict([dense_in3, dense_out3])
mob_time[27]=mob_time27
print(mob_time[27])

mob_time28 = mob_time27+mobile_profiler2.predict([conv_out1])
mob_time[28]=mob_time28
print(mob_time[28])


mob_time29 = mob_time28+mobile_profiler3.predict([conv_out1])
mob_time[29]=mob_time29
print(mob_time[29])
