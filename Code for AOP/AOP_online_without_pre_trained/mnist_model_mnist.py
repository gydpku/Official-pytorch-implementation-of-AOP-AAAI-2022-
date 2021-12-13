from abc import *
import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim=10, num_classes=10, simclr_dim=32*4):
        super(BaseModel, self).__init__()
        #self.linear = nn.Linear(last_dim, num_classes)
        self.out_num= 10
       # self.weight3 = nn.Parameter(torch.Tensor(self.out_num, 300))
        #self.simclr_layer = nn.Sequential(
         #   #nn.Linear(last_dim, last_dim),
          #  nn.ReLU(inplace=True),
           # nn.Linear(last_dim, simclr_dim),
        #)
        #self.shift_cls_layer = nn.Sequential(nn.Linear(last_dim, last_dim),
         #   nn.ReLU(inplace=True),nn.Linear(last_dim, 4))
      #  self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)
        self.fc1 = torch.nn.Linear(28*28, 1000, bias=False)
        self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        self.fc3 = torch.nn.Linear(1000, self.out_num, bias=False)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs, penultimate=False, is_simclr=False, shift=False):
        _aux = {}
        #print("a")
        _return_aux = False
        h_list=[]
        features = self.penultimate(inputs)#这里是MLP最后一层
        h = features
        #print("feature",features.shape)\
       # h=torch.cat([h,torch.ones(h.shape[0],1).cuda()],dim=1)
        h_list.append(torch.mean(h, 0, True).detach())
        h = self.relu(self.fc1(h))
        #features = h
        #print("fea")
        #h = torch.cat([h, torch.ones(h.shape[0], 1).cuda()], dim=1)
        #h_list.append(torch.mean(h, 0, True).detach())
        #h = self.relu(self.fc2(h))
        #features=h
        #print("h")
        #h_list.append(torch.mean(h, 0, True))
        #h = self.fc2(h)
        h_list.append(torch.mean(h, 0, True).detach())
        h = self.relu(self.fc2(h))
        # features = h
        # print("fea")
        # h = torch.cat([h, torch.ones(h.shape[0], 1).cuda()], dim=1)
        h_list.append(torch.mean(h, 0, True).detach())
        output = self.fc3(h)
        #features=output
        #features=h
        del h
        #output=0

        #output = F.linear(features,self.weight3)#再跑一个线性层，变成分类任务



        if _return_aux:
            del features
            return output

        return output,h_list


class Net(BaseModel):

    def __init__(self, inputsize):
        super(Net, self).__init__()
        ncha, size, _ = inputsize
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.padding = torch.nn.ReplicationPad2d(1)

        #self.c1 = torch.nn.Conv2d(ncha, 64, kernel_size=2, stride=1, padding=0, bias=False)
        #self.c2 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=1,  padding=0, bias=False)
        #self.c3 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0, bias=False)
        #self.c4 = torch.nn.Conv2d(128, 512, kernel_size=2, stride=1, padding=0, bias=False)

        #self.fc1 = torch.nn.Linear(256 * 4 * 4, 1000, bias=False)
        #self.fc2 = torch.nn.Linear(1000, 1000, bias=False)
        #self.fc3 = torch.nn.Linear(1000, 10,  bias=False)

        #torch.nn.init.xavier_normal_(self.fc1.weight)
        #torch.nn.init.xavier_normal_(self.fc2.weight)
        #torch.nn.init.xavier_normal_(self.fc3.weight)

        return

    def penultimate(self, x):
        h_list = []
        x_list = []
        # Gated
        #x = self.padding(x)
        #x_list.append(torch.mean(x, 0, True))
        #con1 = self.drop1(self.relu(self.c1(x)))
        #con1_p = self.maxpool(con1)

        #con1_p = self.padding(con1_p)
        #x_list.append(torch.mean(con1_p, 0, True))
        #con2 = self.drop1(self.relu(self.c2(con1_p)))
        #con2_p = self.maxpool(con2)

        #con2_p = self.padding(con2_p)
        #x_list.append(torch.mean(con2_p, 0, True))
        #con3 = self.drop1(self.relu(self.c3(con2_p)))
        #con3_p = self.maxpool(con3)

        h = x.view(x.size(0), -1)

        return h
