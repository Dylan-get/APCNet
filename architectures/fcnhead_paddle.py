import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from . import resnet_paddle
# import resnet_paddle
import warnings
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=False,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                   
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                    pass
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class ConvModule(nn.Layer):
    def __init__(self,in_channels,out_channels,kernel_size=1,padding=0,dilation=1,conv_cfg=None,norm_cfg='bn',act_cfg='relu'):
        super(ConvModule, self).__init__()
        self.conv=nn.Conv2D(in_channels,out_channels,kernel_size=kernel_size,padding=padding,dilation=dilation, bias_attr=False)
        self.bn=nn.BatchNorm2D(out_channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return self.relu(x)
        
class FCNHead(nn.Layer):
    def __init__(self,
                 num_convs=1,
                 kernel_size=3,
                 concat_input=False,
                 dilation=1,
                 dropout_ratio=0.1
                 ):
        super(FCNHead, self).__init__()
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2D(256,19, kernel_size=1)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        
        
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                1024,
                256,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                ))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        # print('FCN head',x.shape) #[2, 1024, 64, 128]
        output = self.convs(x)
        output = self.cls_seg(output)
        # print('fcn output.shape',output.shape) #[2, 19, 64, 128]
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
if __name__=='__main__':
    # models={}
    # models,msg1=resnet101()
    model=FCNHead()
    x=paddle.normal(shape=[2, 1024, 64, 128])
    out=model(x)
    print(out.shape)
   
    