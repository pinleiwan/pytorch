import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

#按住 ctrl 选中 CIFAR10可以跳转到函数定义
datasets = torchvision.datasets.CIFAR10("../data",
                                        train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True,
                                        )

dataloader = DataLoader(datasets,batch_size=64,shuffle=True)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=Linear(64*3*32*32,10)
    def forward(self,input):
        output =self.linear1(input)
        return output

for data in dataloader:
    imgs,labels =data   #[64,3,32,32]
    print(imgs.shape)
    output = torch.reshape(imgs,(1,1,1,-1))
    print(output.shape)
    myModel=Model()
    output=myModel(output)  #会调用forward函数
    print(output)