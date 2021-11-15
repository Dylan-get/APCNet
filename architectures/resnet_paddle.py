import paddle
import torchvision
import torch
import paddle.nn as nn
import numpy as np
import os
def trans():
    path='./resnet_pytorch.pth'
    model=torchvision.models.resnet101(num_classes=1000)
    torch_dict=torch.load(path)
    model.load_state_dict(torch_dict['net'])
    paddle_dict = {}
    # paddleModel=paddle.vision.models.resnet101()
    for key in torch_dict['net']:
        weight=torch_dict['net'][key].cpu().detach().numpy()  
        paddle_dict[key]=weight

    # paddle.save(paddle_dict,'./resnet_paddle.pdparams')

def seePaddle():
    model=paddle.vision.models.resnet101()
    for ele in model.state_dict():
        print(ele)
    
def seePytorch():
    path='./resnet_pytorch.pth'
    torch_dict=torch.load(path)
    model=torchvision.models.resnet101(num_classes=1000)
    model.load_state_dict(torch_dict['net'])
    for ele in model.state_dict():
        print(ele)

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 kernel_size,
                 stride=1,
                 padding=0,
                 act='relu',
                 bias_attr=None):
        super(ConvBNLayer,self).__init__()
        self._conv=nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr
        )
        self._batch_norm=nn.BatchNorm2D(num_filters)
        self.act=act
    def forward(self,inputs):
        x=self._conv(inputs)
        x=self._batch_norm(x)
        if self.act=='leaky':
            x=nn.functional.leaky_relu(x=x,negative_slope=0.1)
        elif self.act=='relu':
            x=nn.functional.relu(x=x)
        return x
class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1,dilation=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1,stride=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        # print('dilation:',dilation)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,dilation=dilation,padding=dilation, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
 
        return out
class ResNet(nn.Layer):
    def __init__(self, block, layers, num_classes=1000,type='c'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2D(7, stride=1)

        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #[1, 2048, 24, 24]
        return x
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed."""
    #     super(ResNet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()
class ResNetV1C(nn.Layer):
    def __init__(self, block=Bottleneck, layers=[3, 4, 23, 3]):
        self.inplanes = 64
        super(ResNetV1C, self).__init__()
        self.dilations = (1, 1, 2, 4)
        self.strides=(1, 2, 1, 1)
        self.conv1 = nn.Sequential(
                            ConvBNLayer(3, 32, kernel_size=3, stride=2,padding=1,act='relu',bias_attr=False),
                            ConvBNLayer(32, 32, kernel_size=3, stride=1,padding=1,act='relu',bias_attr=False),
                            ConvBNLayer(32, 64, kernel_size=3, stride=1,padding=1,act='relu',bias_attr=False),
                            )

        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],stride=1,dilation=1)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilation=1)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,dilation=2)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation=4)

        self.num_features = 512 * block.expansion

        
        
    def _make_layer(self, block, planes, blocks, stride=1,dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )
            
        layers = []
        layers.append(block(self.inplanes, planes,stride=stride,downsample=downsample))
    
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,stride=1,dilation=dilation))

        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        # print('resnet paddel stem',x.shape) #should be[2, 64, 256, 512]
        x = self.maxpool(x)
        # print('resnet  paddel maxpool',x.shape) #[2, 64, 128, 256]
        x = self.layer1(x)
        # print(x.shape) #[2, 256, 128, 256]
        x = self.layer2(x)
        # print(x.shape) #[2, 512, 64, 128]
        x = self.layer3(x)
        # print(x.shape) #[2, 1024, 64, 128]
        tmp=x
        x = self.layer4(x)
        # print(x.shape) #[2, 2048, 64, 128]
        # x = self.avgpool(x)
        # x = x.reshape([x.shape[0],-1])
        #[1, 2048, 24, 24]
        """
        resnet torch layer 0  : torch.Size([2, 256, 128, 256])
        resnet torch layer 1  : torch.Size([2, 512, 64, 128])
        resnet torch layer 2  : torch.Size([2, 1024, 64, 128])
        resnet torch layer 3  : torch.Size([2, 2048, 64, 128])
        """
        return [tmp,x]

def resnet101(pretrained=True,config=None):
    print('-vghjguiio')
    model = ResNetV1C(Bottleneck, [3, 4, 23, 3])
    print('-gdsfgsdfgdssdgf')
    if pretrained:
        path=config['model']['resnet']['resnet_pretrained_root']
        state=paddle.load(path)
        model.set_state_dict(state['state_dict'])
        msg='----resnet101 pretrained load success from {}----'.format(path)
        # model.set_dict(param)
    else:
        msg='----resnet101 pretrained load FALSE!!!----'
    return model,msg

if __name__=='__main__':
    # models={}
    # models,msg1=resnet101()
    model,msg=resnet101()
    # print(model)
    # paddle.summary(model)
    # out=paddle.summary(model, (2, 3,512,1024)) #[4, 2048, 1, 1]
    # print(out.shape)
    #inpu [8, 3, 512, 1024]
    # 
    #[2, 2048, 64, 128]
    x=paddle.normal(shape=[2,3,512,1024])
    out=model(x)
    # print(out.shape)
    # print(model.keys())
    # model=resnet101()
    # data_1 = np.random.rand(1, 3, 768, 768).astype(np.float32)
    # data_1=paddle.to_tensor(data_1)
    # out=model(data_1)
    # print('----')
    # print(out.shape)
    # seePaddle()
    # print('----')
    # seePytorch()