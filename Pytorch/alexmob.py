import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import socket
import pickle
import json
from timeit import default_timer as timer



images = torch.zeros(9,3,224,224)


original_model = models.alexnet(pretrained=True)

print("Main model")
child_counter = 0
for child in original_model.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1


class AlexNetConv4(nn.Module):
            def __init__(self):
                super(AlexNetConv4, self).__init__()
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(original_model.features.children())[:-3]
                )
            def forward(self, x):
                x = self.features(x)
                return x

model1 = AlexNetConv4()
print("Mobile model")
child_counter = 0
for child in model1.children():
   print(" child", child_counter, "is:")
   print(child)
   child_counter += 1

outputs1 = model1(images)
print(outputs1)

print("Sending Data to server")
start=timer()

TCP_IP2 = '129.32.94.246'
TCP_PORT2 = 5006

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP2,TCP_PORT2))
send_x=outputs1.detach().numpy()
data_input=pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
s.sendall(data_input)
s.close

print("data sent to server")


#print('Final Output')
TCP_IP2 = '10.108.27.196'
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
print('Output shown on mobile side')
end=timer()
conn.close
print("final time =")
print (end - start)
     #conn.send(data)





























