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
                                    Convolution2D, MaxPooling2D, BatchNormalization, ReLU
                                  

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


#Raspberry Pi Profiler


serv_time=np.zeros(num_of_partitions)


Server_profiler1 = Profiler('CNN')
Server_profiler2 = Profiler('BatchNormalization')
Server_profiler4 = Profiler('Pooling')
Server_profiler3 = Profiler('Relu')
#Server_profiler5 = Profiler('Dropout')
Server_profiler7 = Profiler('Dense')

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

conv_layer1=Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
conv_layer2=Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
conv_layer3=Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
conv_layer4=Server_profiler1.predict([conv_in4, conv_out4])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
conv_layer5=Server_profiler1.predict([conv_in5, conv_out5])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
dense_layer1=Server_profiler7.predict([dense_in1, dense_out1])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])
dense_layer2=Server_profiler7.predict([dense_in2, dense_out2])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])
dense_layer3=Server_profiler7.predict([dense_in3, dense_out3])+Server_profiler2.predict([conv_out1])+Server_profiler3.predict([conv_out1])
serv_time0 =conv_layer1+conv_layer2+conv_layer3+conv_layer4+conv_layer5+dense_layer1+dense_layer2+dense_layer3
serv_time[0]=serv_time0
print(serv_time[0])

serv_time1 = serv_time0-Server_profiler1.predict([conv_in1, conv_out1])
serv_time[1]=serv_time1
print(serv_time[1])

serv_time2 = serv_time1-Server_profiler2.predict([conv_out1])
serv_time[2]=serv_time2
print(serv_time[2])

serv_time3 = serv_time2-Server_profiler3.predict([conv_out1])
serv_time[3]=serv_time3
print(serv_time[3])

serv_time4 = serv_time3-Server_profiler4.predict([pool_in1,pool_out1])
serv_time[4]=serv_time4
print(serv_time[4])


serv_time5 = serv_time4-Server_profiler1.predict([conv_in2, conv_out2])
serv_time[5]=serv_time5
print(serv_time[5])

serv_time6 = serv_time5-Server_profiler2.predict([conv_out1])
serv_time[6]=serv_time6
print(serv_time[6])

serv_time7 = serv_time6-Server_profiler3.predict([conv_out1])
serv_time[7]=serv_time7
print(serv_time[7])

serv_time8 = serv_time7-Server_profiler4.predict([pool_in1,pool_out1])
serv_time[8]=serv_time8
print(serv_time[8])

serv_time9 = serv_time8-Server_profiler1.predict([conv_in3, conv_out3])
serv_time[9]=serv_time9
print(serv_time[9])

serv_time10 = serv_time9-Server_profiler2.predict([conv_out1])
serv_time[10]=serv_time10
print(serv_time[10])

serv_time11 = serv_time10-Server_profiler3.predict([conv_out1])
serv_time[11]=serv_time11
print(serv_time[11])

serv_time12 = serv_time11-Server_profiler4.predict([pool_in1,pool_out1])
serv_time[12]=serv_time12
print(serv_time[12])


serv_time13 = serv_time12-Server_profiler1.predict([conv_in4, conv_out4])
serv_time[13]=serv_time13
print(serv_time[13])

serv_time14 = serv_time13-Server_profiler2.predict([conv_out1])
serv_time[14]=serv_time14
print(serv_time[14])

serv_time15 = serv_time14-Server_profiler3.predict([conv_out1])
serv_time[15]=serv_time15
print(serv_time[15])

serv_time16 = serv_time15-Server_profiler4.predict([pool_in1,pool_out1])
serv_time[16]=serv_time16
print(serv_time[16])

serv_time17 = serv_time16-Server_profiler1.predict([conv_in5, conv_out5])
serv_time[17]=serv_time17
print(serv_time[17])

serv_time18 = serv_time17-Server_profiler2.predict([conv_out1])
serv_time[18]=serv_time18
print(serv_time[18])

serv_time19 = serv_time18-Server_profiler3.predict([conv_out1])
serv_time[19]=serv_time19
print(serv_time[19])

serv_time20 = serv_time19-Server_profiler4.predict([pool_in1,pool_out1])
serv_time[20]=serv_time20
print(serv_time[20])

serv_time21 = serv_time20-Server_profiler7.predict([dense_in1, dense_out1])
serv_time[21]=serv_time21
print(serv_time[21])

serv_time22 = serv_time21-Server_profiler2.predict([conv_out1])
serv_time[22]=serv_time22
print(serv_time[22])

serv_time23 = serv_time22-Server_profiler3.predict([conv_out1])
serv_time[23]=serv_time23
print(serv_time[23])

serv_time24 = serv_time23-Server_profiler7.predict([dense_in2, dense_out2])
serv_time[24]=serv_time24
print(serv_time[24])

serv_time25 = serv_time24-Server_profiler2.predict([conv_out1])
serv_time[25]=serv_time25
print(serv_time[25])

serv_time26 = serv_time25-Server_profiler3.predict([conv_out1])
serv_time[26]=serv_time26
print(serv_time[26])

serv_time27 = serv_time26-Server_profiler7.predict([dense_in3, dense_out3])
serv_time[27]=serv_time27
print(serv_time[27])

serv_time28 = serv_time27-Server_profiler2.predict([conv_out1])
serv_time[28]=serv_time28
print(serv_time[28])


print(serv_time[29])
