import tensorflow as tf
import os
import tensorflow.contrib.eager as tfe
import numpy as np

import time
import matplotlib.pyplot as plt
import socket
import pickle

from tensorflow.python.training import checkpointable
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
                                    Convolution2D, MaxPooling2D, BatchNormalization, ReLU

#eager execution enabled
tf.enable_eager_execution()

#importing profiler
#from profiler import Profiler


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
num_of_partitions =22



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
#def execution_time():
  #rt = np.zeros(num_of_partitions)
  #for bwn in range(bw_data):
  #for inr in range(num_of_partitions):
      #rt[inr]=(serv_time[inr]+mob_time[inr])
      #rt[inr]=(serv_time[inr]+mob_time[inr]+input_rate[0])
  #return rt



#model building 

model = MyModel.AlexNet()

#calculating input rate
#data = 10 #kb
#bandwidth=np.arange(50,550,100)

#input_rate = data/bandwidth

#input_rate=([0,2,1.2,0.1,0.2,0.5,0.3,0])

#bw_data = np.size(input_rate)

#partition_num=np.zeros([1,bw_data])

#latency_user=4.5

#bw_wonder_start=0

#algorithm for partition point


#serv_time=np.zeros(num_of_partitions)
#mob_time=([4.1454,2.7693,1.3885,0.0183,0.0113,0.0054])


#Server_profiler1 = Profiler('CNN')
#Server_profiler2 = Profiler('BatchNormalization')
#Server_profiler4 = Profiler('Pooling')
#Server_profiler3 = Profiler('Relu')
#Server_profiler5 = Profiler('Dropout')
#Server_profiler6 = Profiler('Dense')
#Server_profiler7 = Profiler('Dense')  

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
#dense_out1=20
#dense_in2=10
#dense_out2=20
#dense_in3=10
#dense_out3=20



#serv_time0 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler7.predict([dense_in3, dense_out3])+Server_profiler7.predict([dense_in1, dense_out1])+Server_profiler7.predict([dense_in2, dense_out2])
#serv_time[0]=serv_time0
#print(serv_time[0])

#serv_time1 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler7.predict([dense_in1, dense_out1])+Server_profiler7.predict([dense_in2, dense_out2])
#serv_time[1]=serv_time1
#print(serv_time[1])

#serv_time2 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler7.predict([dense_in1, dense_out1])
#serv_time[2]=serv_time2
#print(serv_time[2])

#serv_time3 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
#serv_time[3]=serv_time3
#print(serv_time[3])

#serv_time4 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])+Server_profiler3.predict([conv_out1])
#serv_time[4]=serv_time4
#print(serv_time[4])

#serv_time5 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in3, conv_out3])
#serv_time[5]=serv_time5
#print(serv_time[5])

#serv_time6 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
#serv_time[6]=serv_time6
#print(serv_time[6])

#serv_time7 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])+Server_profiler3.predict([conv_out1])
#serv_time[7]=serv_time7
#print(serv_time[7])

#serv_time8 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])+Server_profiler1.predict([conv_in2, conv_out2])
#serv_time[8]=serv_time8
#print(serv_time[8])

#serv_time9 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])+Server_profiler4.predict([pool_in1,pool_out1])
#serv_time[9]=serv_time9
#print(serv_time[9])

#serv_time10 = Server_profiler1.predict([conv_in1, conv_out1])+Server_profiler3.predict([conv_out1])
#serv_time[10]=serv_time10
#print(serv_time[10])

#serv_time11 = Server_profiler1.predict([conv_in1, conv_out1])
#serv_time[11]=serv_time11
#print(serv_time[11])

#print(serv_time[12])

x=tf.random_normal([1,427,427,3])
inputs=tf.contrib.eager.Variable(x,name='weights')
y = model(inputs)
model.summary()






#receiving data
while 1:
 print('Waiting for Raspberry 1')

 #bw_wonder_dl=bw_wonder_start+50
 #bw_wonder_ul=((bw_wonder_start+50)/2)
 #os.system('echo aa | sudo -S wondershaper enp0s31f6 {} {}'.format(bw_wonder_dl,bw_wonder_ul))

#print('server execution times:')
#print(serv_time)

 

 #client=paramiko.SSHClient()
 #client.load_system_host_keys()
 #client.connect('192.168.1.55',22,username='pi',password='raspberry1')
 #print('Connected to rasberry pi 1')
 #stdin,stdout,stderr=client.exec_command('python /home/pi/tf1/mobile_server_partition.py')
 #print('Script executed in RPI1')
 #client.close()

 



#serv_time=np.random.rand(1,num_of_layers)

#mob_time=np.random.rand(1,num_of_layers)



#output_rate=np.random.rand(1,8)

#tensorflow serving for mob time


#for bw in range(bw_data):

 #print('Current Bandwidth:')
 #print(bandwidth[bw])

 #run_time1 = execution_time()
 #final_rt=run_time1 + input_rate[bw]
 
 #print('Total Run time:')
 #print(final_rt)

#partition point
 
 #minimum_let=min(final_rt)

 #for j in range(num_of_partitions):
   #if final_rt[j]==minimum_let:
    #part_pnt=j+1
    #break
	    
   #else:
 TCP_IP1 ='192.168.1.119'
 TCP_PORT1 = 5006
 BUFFER_SIZE = 4096



 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.bind((TCP_IP1, TCP_PORT1))
 print('Waiting for partition point')
 s.listen(2)
 
 
 conn, addr = s.accept()
 print ('Raspberry Device:',addr)
 while 1:
  data = conn.recv(BUFFER_SIZE)
  if not data: break
  part_pnt=pickle.loads(data)
   
 print('\n'+"Partition point: ")
 print(part_pnt)

#os.system('sudo wondershaper remove enp0s31f6')
#print(bandwidth)
#part_num=partition_num.reshape((bw_data,))
#print(part_num)
#plt.plot(bandwidth,part_num)
#plt.xlabel('Bandwidth(kbps)') 
#plt.ylabel('Partition Points')
#plt.axis([0,2000,0,6])
#plt.show()

 if part_pnt<=0:
  print('waiting for input')
  conn, addr = s.accept()
  #start1=time.time()
  data=[]
  print ('Raspberry Device:',addr)
  while 1:
   tensor = conn.recv(BUFFER_SIZE)
   if not tensor: break
   data.append(tensor)

  inputs_ten=pickle.loads(b"".join(data))
  
   #conn.send(data)
  conn.close()
  inputs=tf.convert_to_tensor(inputs_ten)
  input_size=inputs.shape
  #print(inputs)
  #print(input_size)
  
  
  final_outputs = server_call(model, inputs, part_pnt)
  final_output_send =final_outputs.numpy()
  #print('server output')
  print(final_output_send)
  TCP_IP2 ='192.168.1.55'
  TCP_PORT2 = 5005
  BUFFER_SIZE = 4096

  s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((TCP_IP2, TCP_PORT2))
 
 #filename = 'final_output.npy'
 #outfile = open(filename,'wb')
  data_final=pickle.dumps(final_output_send,protocol=pickle.HIGHEST_PROTOCOL)

 #time.sleep(1)
  s.sendall(data_final)

 #data=s.recv(BUFFER_SIZE)
 #data_arr=pickle.loads(data)
  s.close()

  print('Server execution done!!')
  #end1=time.time()
  #print('Total execution time: ',end1-start1)
#saver = tfe.Saver

#input data

#x = tf.ones([10, 50, 50, 3])
#x=tf.random_normal([10,50,50,3], stddev=0.2)
#inputs = tfe.Variable(x, name='weights') 

#partition data

#partition_outputs = mobile_call(model, inputs, part_pnt)
#print(partition_outputs)
  

 elif part_pnt<num_of_partitions-1:
  print('waiting for partial output')
  
  conn, addr = s.accept()
  data=[]
  print ('Raspberry Device:',addr)
  while 1:
   tensor = conn.recv(BUFFER_SIZE)
   if not tensor: break
   data.append(tensor)

  part_outputs=pickle.loads(b"".join(data))
 
   #conn.send(data)
  conn.close()
  
 
  partition_outputs = tf.convert_to_tensor(part_outputs)
  print(partition_outputs)
 
 
  final_outputs = server_call(model, partition_outputs, part_pnt)
  final_output_send =final_outputs.numpy()
  print('server output')
  print(final_output_send)
 #print(final_output_send.shape)
 #print(final_output_send.size)
 
  TCP_IP2 ='192.168.1.55'
  TCP_PORT2 = 5005
  BUFFER_SIZE = 4096



  s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect((TCP_IP2, TCP_PORT2))
 
 #filename = 'final_output.npy'
 #outfile = open(filename,'wb')
  data_final=pickle.dumps(final_output_send,protocol=pickle.HIGHEST_PROTOCOL)

 #time.sleep(1)
  s.sendall(data_final)

 #data=s.recv(BUFFER_SIZE)
 #data_arr=pickle.loads(data)
  s.close()

  print('Server execution done!!')
 
 else:
  
  print('NO SERVER EXECUTION!!')
  


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
 

















