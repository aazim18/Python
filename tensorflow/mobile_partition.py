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

#calculating input rate
data_layer = ([44515584,44515584,44515584,4946176,9056768,9056768,9056768,991232,1354752,1354752,1354752,150528,200704,200704,200704,16384,16384,16384,16384,1024,16384,16384,16384,16384,16384,16384,4000,4000,4000,0])
#bandwidth=np.arange(50,550,100)
bandwidth=50000#bytes/sec

input_rate = [data/bandwidth for data in data_layer]

#input_rate=([0.05, 0.1, 0.15,0.25,0.4,0.5,2, 9, 15, 35.2, 38, 40, 48])
#input_rate=([48,40, 38, 35.2, 15, 9, 5.2, 0.5, 0.4,0.25,0.15, 0.1, 0.05])
#input_rate=([8.65, 8.64, 8.63,6.54, 6.65, 6.46,6.44, 4.25, 4.15,4.11, 3.89, 2.45, 2.2,2.19,2.15,1.2,1.1,0.9,0.7,0.3,0.25,0.22,0.21,0.18,0.16,0.15,0.11,0.09,0.05,0])

#bw_data = np.size(input_rate)

#partition_num=np.zeros([1,bw_data])


#Raspberry Pi Profiler

serv_time=([0.7227,0.5853,0.5818,0.5816,0.5815,0.4441,0.44059,0.4404,0.4403,0.3029,0.2994,0.2992,0.299,0.1616,0.1581,0.1579,0.1578,0.0204,0.0169,0.0167,0.0166,0.0148,0.0113,0.0111,0.0092,0.0057,0.0055,0.0037,0.00019,0])
#mob_time=np.zeros(num_of_partitions)
mob_time=([0,3.2139,3.2925,3.2940,3.2954,6.5094,6.588,6.5894,6.5909,9.8049,9.8835,9.8849,9.8863,13.1003,13.1789,13.1803,13.1818,16.3958,16.4743,16.4757,16.4772,16.515,16.5935,16.595,16.632,16.7113,16.7128,16.7505,16.829,16.8305])


#Server_profiler1 = Profiler('CNN')
#Server_profiler2 = Profiler('CNN')
#Server_profiler3 = Profiler('CNN')
#Server_profiler4 = Profiler('Dense')  

#mobile_profiler1 = Profiler('CNN')
#mobile_profiler2 = Profiler('BatchNormalization')
#mobile_profiler4 = Profiler('Pooling')
#mobile_profiler3 = Profiler('Relu')
#mobile_profiler5 = Profiler('Dense')
#mobile_profiler6 = Profiler('Dropout')
#mobile_profiler7 = Profiler('Dense')

#serv_time1 = Server_profiler1.predict([120,50])
#serv_time[0]=serv_time1
#print(serv_time1)

#conv_in1=120 
#conv_out1=50
#pool_in1=120
#pool_out1=50
#conv_in2=120 
#conv_out2=50
#conv_in3=120 
#conv_out3=50
#conv_in4=120 
#conv_out4=50
#dense_in1=10
#conv_in5=120 
#conv_out5=50
#dense_out1=20
#dense_in2=10
#dense_out2=20
#dense_in3=10
#dense_out3=20

#print(mob_time[0])

#mob_time1 = mobile_profiler1.predict([conv_in1, conv_out1])
#mob_time[1]=mob_time1
#print(mob_time[1])

#serv_time2 = Server_profiler1.predict([120,50])+Server_profiler2.predict([120,50])
#serv_time[1]=serv_time2
#print(serv_time2)

#mob_time2 = mob_time1+mobile_profiler3.predict([conv_out2])


#mob_time[2]=mob_time2
#print(mob_time[2])


#mob_time3 = mob_time2+mobile_profiler4.predict([pool_in1,pool_out1])
#mob_time[3]=mob_time3
#print(mob_time[3])

#serv_time3 = Server_profiler1.predict([120,50])+Server_profiler2.predict([120,50])+Server_profiler3.predict([120,50])
#serv_time[2]=serv_time3
#print(serv_time3)

#mob_time4 = mob_time3 +mobile_profiler1.predict([conv_in2, conv_out2])

#mob_time[4]=mob_time4
#print(mob_time[4])

#mob_time5 = mob_time4+mobile_profiler3.predict([conv_out1])
#mob_time[5]=mob_time5
#print(mob_time[5])

#mob_time6 = mob_time5+mobile_profiler4.predict([pool_in1,pool_out1])
#mob_time[6]=mob_time6
#print(mob_time[6])

#mob_time7 = mob_time6+mobile_profiler1.predict([conv_in3, conv_out3])
#mob_time[7]=mob_time7
#print(mob_time[7])

#mob_time8 = mob_time7+mobile_profiler3.predict([conv_out1])
#mob_time[8]=mob_time8
#print(mob_time[8])

#mob_time9 = mob_time8+mobile_profiler4.predict([pool_in1,pool_out1])
#mob_time[9]=mob_time9
#print(mob_time[9])

#mob_time10 = mob_time9+mobile_profiler1.predict([conv_in4, conv_out4])
#mob_time[10]=mob_time10
#print(mob_time[10])

#mob_time11 = mob_time10+mobile_profiler3.predict([conv_out1])
#mob_time[11]=mob_time11
#print(mob_time[11])

#mob_time12 = mob_time11+mobile_profiler4.predict([pool_in1,pool_out1])
#mob_time[12]=mob_time12
#print(mob_time[12])

#mob_time13 = mob_time12+mobile_profiler1.predict([conv_in5, conv_out5])
#mob_time[13]=mob_time13
#print(mob_time[13])

#mob_time14 = mob_time13+mobile_profiler3.predict([conv_out1])
#mob_time[14]=mob_time14
#print(mob_time[14])

#mob_time15 = mob_time14+mobile_profiler4.predict([pool_in1,pool_out1])
#mob_time[15]=mob_time15
#print(mob_time[15])


#mob_time16 = mob_time15+mobile_profiler7.predict([dense_in1, dense_out1])
#mob_time[16]=mob_time16
#print(mob_time[16])

#mob_time17 = mob_time16+mobile_profiler3.predict([conv_out1])
#mob_time[17]=mob_time17
#print(mob_time[17])


#mob_time18 = mob_time17+mobile_profiler7.predict([dense_in2, dense_out2])
#mob_time[12]=mob_time11
#print(mob_time[12])

#mob_time19 = mob_time18+mobile_profiler3.predict([conv_out1])
#mob_time[19]=mob_time19
#print(mob_time[19])

#mob_time20 = mob_time19+mobile_profiler7.predict([dense_in3, dense_out3])
#mob_time[20]=mob_time20
#print(mob_time[20])

#mob_time21 = mob_time20+mobile_profiler3.predict([conv_out1])
#mob_time[21]=mob_time21
#print(mob_time[21])


#serv_time=np.random.rand(1,num_of_layers)

#mob_time=np.random.rand(1,num_of_layers)



#output_rate=np.random.rand(1,8)



#run_time1 = execution_time()

#print(run_time1)



#run_time =(run_time1.reshape(3)) 

#for j in range(num_of_layers):
 #print(run_time[0,j])

#for bw in range(bw_data):

x=tf.random_normal([1,427,427,3], stddev=0.2)
inputs = tf.contrib.eager.Variable(x, name='weights')

#y=model(inputs)
#model.summary()

#c1=tf.random_normal([10,150,150,3], stddev=0.2)
#inputs_c = tf.contrib.eager.Variable(c1, name='weights')
#start_c=time.time()
#conv_output = mobile_call(model, inputs_c, 1)
#end_c=time.time()
#print('Execution time for 1 convolution layer: ',end_c-start_c)
#print('Execution time prediction for 1 convolution layer: ',mob_time1)

#start_d1=time.time()
#dense_output1 = mobile_call(model, inputs_c, 9)
#end_d1=time.time()
#d1=end_d1-start_d1
#start_d2=time.time()
#dense_output2 = mobile_call(model, inputs_c, 10)
#end_d2=time.time()
#d2=end_d2-start_d2
#print('Execution time for 1 Dense layer: ',d2-d1)
#d_time=mobile_profiler7.predict([150,150])
#print('Execution time prediction for 1 Dense layer: ',d_time)
#print('Current Bandwidth:')
#print('500 kbps')
 #part_pnt_send=[]

start1=time.time()

run_time1 = execution_time()

final_rt=run_time1 + input_rate
#final_rt=run_time1
print('Total Run time:')
print(final_rt)

minimum_let=min(final_rt)

for j in range(num_of_partitions):
 if final_rt[j]==minimum_let:
  part_pnt=j
  break
	    
 else:
  part_pnt = 0

#partition_num[0,bw]=part_pnt
print('\n'+"Partition point: ")
print(part_pnt)
part_pnt_send=part_pnt

if part_pnt<=0:
 

 TCP_IP1 = '192.168.1.119'
 TCP_PORT1 = 5006

 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((TCP_IP1,TCP_PORT1))
 data_string1=pickle.dumps(part_pnt_send,protocol=pickle.HIGHEST_PROTOCOL)
 s.send(data_string1)
 s.close
 
 TCP_IP2 = '192.168.1.119'
 TCP_PORT2 = 5006

 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((TCP_IP2,TCP_PORT2))
 send_x=inputs.numpy()
 data_input=pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
 s.sendall(data_input)
 s.close
 
 print('Final Output')


 TCP_IP2 = '192.168.1.55'
 TCP_PORT2 = 5005
 buf_size=4096
 
 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.bind((TCP_IP2,TCP_PORT2))
 s.listen(1)
 
 conn, addr = s.accept()
 data=[]
 print('Server: ',addr)
 
 
 while 1:
    output = conn.recv(buf_size)
    if not output: break
    data.append(output)
 final_output=pickle.loads(b"".join(data))
 print(final_output)
     #conn.send(data)
 conn.close
 print('Output Shown on Mobile side')
 end1=time.time()
 print('Total execution time: ',end1-start1)

elif part_pnt<num_of_partitions-1: 
 TCP_IP1 = '192.168.1.119'
 TCP_PORT1 = 5006

 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((TCP_IP1,TCP_PORT1))
 data_string1=pickle.dumps(part_pnt_send,protocol=pickle.HIGHEST_PROTOCOL)
 s.send(data_string1)
 
 s.close
#print(bandwidth)
#part_num=partition_num.reshape((bw_data,))
#print(part_num)
#plt.plot(bandwidth,part_num)
#plt.xlabel('Bandwidth(kbps)') 
#plt.ylabel('Partition Points')
#plt.axis([0,2000,0,6])
#plt.show()


  
#saver = tfe.Saver


#input data

 #x = tf.ones([10, 50, 50, 3])


#partition data

 partition_outputs = mobile_call(model, inputs, part_pnt)
 #saver_part=np.save('partial_data.npy', partition_outputs.numpy())
 part_output= partition_outputs.numpy()
#print(partition_outputs)

 

 TCP_IP2 = '192.168.1.119'
 TCP_PORT2 = 5006

 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((TCP_IP2,TCP_PORT2))

 data_part=pickle.dumps(part_output, protocol=pickle.HIGHEST_PROTOCOL)

 
 s.sendall(data_part)

 #data=s.recv(4096)
 #data_arr=pickle.loads(data)
 
 #data=s.recv(4096)
 s.close()

 print('Final Output')


 TCP_IP2 = '192.168.1.55'
 TCP_PORT2 = 5005
 buf_size=4096
 
 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.bind((TCP_IP2,TCP_PORT2))
 s.listen(1)
 
 conn, addr = s.accept()
 data=[]
 print('Server: ',addr)
 
 
 while 1:
    output = conn.recv(buf_size)
    if not output: break
    data.append(output)
 final_output=pickle.loads(b"".join(data))
 print(final_output)
     #conn.send(data)
 conn.close
 print('Output Shown on Mobile side')
 end1=time.time()
 print('Total execution time: ',end1-start1)

else:
 
 #TCP_IP1 = '192.168.1.118'
 #TCP_PORT1 = 5006

 #s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 #s.connect((TCP_IP1,TCP_PORT1))
 #data_string1=pickle.dumps(part_pnt_send,protocol=pickle.HIGHEST_PROTOCOL)
 #s.send(data_string1)
 #s.close
 
 start2=time.time()   
 mobile_outputs=mobile_call(model, inputs, part_pnt+1)
 final_output=mobile_outputs.numpy()
 print(final_output)
 end2=time.time()
 print('Output Shown on Mobile side')
 print('Total execution time: ',end2-start2)
 #print('Output from server:')
 #print(final_outpit)
#final_outputs = server_call(model, partition_outputs, part_pnt)

#partial data saved in .numpy file
#saver1 = saver(partition_outputs.numpy())
#saver_data = np.save('partial_data.npy',partition_outputs.numpy())

#saver1.save('partialdata.npy')


#saver1.save('./model.ckpt')
#saver1.restore('model.ckpt')

#checkpoint = tfe.Checkpoint(model=model)

#checkpoint = tfe.Checkpoint(model=Sequential(model.layers[:4]))

#saver2 = tfe.CheckpointableSaver(checkpoint)
#saver2.save('./check_model.ckpt')
#saver2.restore('check_model.ckpt')


#print(saver1)
#print('\n')
#print(saver2)


#distributed tensorflow

#device0='/job:node/task:0'

#device1='/job:node/task:1'

#devices=(device0,device1)

#cluster_spec = tf.train.ClusterSpec({'node': [('192.168.1.117'+":"+'2222'),('192.168.1.55'+":"+'2224')]})

#task_idx=0
#server = tf.train.Server(cluster_spec, job_name='node',task_index=task_idx)
#server.join()

#with tf.device(devices[0]):
 #saver3=saver1

#with tf.device(devices[1]):
 #saver4=saver2

#with tf.Session(server.target) as sess: 
 #print(saver3)
 #print('\n')
 #print(saver4)
 

















