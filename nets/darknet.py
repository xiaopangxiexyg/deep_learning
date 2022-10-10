import math
from collections import OrderedDict #有序

import torch.nn as nn


#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes): # net的初始化函数，定义了神经网络的基本结构
        super(BasicBlock, self).__init__() # 复制并使用net父类的初始化方法，即先运行nn.Module的初始化函数
        self.conv1  = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(planes[0]) # 数据的归一化处理，常用于卷积之后。使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.relu1  = nn.LeakyReLU(0.1) # 数据修正，Leaky ReLU是给所有负值赋予一个非零斜率
        
        self.conv2  = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(planes[1])
        self.relu2  = nn.LeakyReLU(0.1)

    def forward(self, x): # net的向前传播函数，向后传播会自动生成
        residual = x # 残差边

        out = self.conv1(x) #主干边 
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual # 残差边与主干边在尾部相连
        return out

class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32 初始化
        self.conv1  = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) # 32通道卷积
        self.bn1    = nn.BatchNorm2d(self.inplanes) # 标准化
        self.relu1  = nn.LeakyReLU(0.1) # 激活函数

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0]) # 残差块 卷积核大小、堆叠层数 1
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1]) # 2
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2]) # 8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3]) # 8
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4]) # 4

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        ##Fn+Ctrl,更换功能键
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False))) # 下采样，输入的特征层长宽减小，通道数增加
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks): # blocks是残差块堆叠次数
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x) # 后三个结构块处理后的结果，之后要进行处理
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53():
    model = DarkNet([1, 2, 8, 8, 4]) #残差块Residual Block使用次数
    return model
