from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
import torch
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2,stride=1) #(in_channels, out_channels, kernel_size,padding=2, stride=1)
        self.maxpool1=MaxPool2d(2)
        self.conv2=Conv2d(32,32,5,padding=2,stride=1)
        self.maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(32,64,5,padding=2,stride=1)
        self.maxpool3=MaxPool2d(2)
        self.flatten=Flatten()
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)
    def forward(self,x):                    #CAFAR10的输入维度，(N,3,32,32), N为batch_size=64
        x = self.conv1(x)                   # x = (N,32,32,32)  (32+2*2-5)/1+1=32    padding=2 s=1 kernel_size=5
        x = self.maxpool1(x)                # x = (N,32,16,16)  32/2=16
        x = self.conv2(x)                   # x = (N,32,16,16)  (16+2*2-5)/1+1=16
        x = self.maxpool2(x)                # x = (N,32,8,8)    16/2=8
        x = self.conv3(x)                   # x = (N,64,8,8)    (8+2*2-5)/1+1=8
        x = self.maxpool3(x)                # x = (N,64,4,4)    8/2=4
        x = self.flatten(x)                 # x = (N,1024)      64*4*4=1024
        x = self.linear1(x)                 # x = (N,64)
        x = self.linear2(x)                 # x = (N,10)
        return x
#Sequential  方式
# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.model1=Sequential(
#             Conv2d(3, 32, 5, padding=2,stride=1),  # (in_channels, out_channels, kernel_size,padding=2, stride=1)
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2, stride=1),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2, stride=1),
#             MaxPool2d(2),
#             Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self,x):
#         x = self.model1(x)
#         return x

myModel=Model()
# print(myModel)

#验证网路搭建的对不对
input  =torch.ones((64,3,32,32))  #ctrl + p 查看参数
output=myModel(input)
print(output.shape)





writer = SummaryWriter("../logs_seq")
writer.add_graph(myModel,input)
writer.close()