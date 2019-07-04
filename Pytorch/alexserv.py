import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import socket
import pickle
import json

batch_size = 100

images = torch.zeros(9,3,224,224)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

original_model = models.alexnet(pretrained=True)

print("Main model")
child_counter = 0
for child in original_model.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1


class AlexNetConv5(nn.Module):
            def __init__(self):
                super(AlexNetConv5, self).__init__()
                self.features = nn.Sequential(
                    # start at conv5
 		    *(list(original_model.features.children())[-3:] + [nn.AvgPool2d(1), Flatten()] + list(original_model.classifier.children()))
                        
                 )
            def forward(self, x):
                x = self.features(x)
                return x

model2 = AlexNetConv5()
print("Server model")
child_counter = 0
for child in model2.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

#receiving data
while 1:
 print('Waiting for Raspberry 1')
 
 TCP_IP1 ='129.32.94.251'
 TCP_PORT1 = 5006
 BUFFER_SIZE = 4096
 
 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.bind((TCP_IP1, TCP_PORT1))
 s.listen(1)

 conn, addr = s.accept()
 data=[]
 print ('Raspberry Device:',addr)
 while 1:
  tensor = conn.recv(BUFFER_SIZE)
  if not tensor: break
  data.append(tensor)
 inputs_ten=pickle.loads(b"".join(data))
  
 conn.close()
 inputs=torch.from_numpy(inputs_ten)



 outputs2 = model2(inputs)
 final_output_send =outputs2.numpy()

 print(final_output_send)
 TCP_IP2 ='10.108.31.196'
 TCP_PORT2 = 5005
 BUFFER_SIZE = 4096

 s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((TCP_IP2, TCP_PORT2))
 data_final=pickle.dumps(final_output_send,protocol=pickle.HIGHEST_PROTOCOL)
 s.sendall(data_final)
 print('Server execution done!!')





