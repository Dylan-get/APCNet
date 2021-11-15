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
    def __init__(self,in_channels,out_channels,kernel_size=1,padding=0,conv_cfg=None,norm_cfg='bn',act_cfg='relu'):
        super(ConvModule, self).__init__()
        self.conv=nn.Conv2D(in_channels,out_channels,kernel_size=kernel_size,padding=padding, bias_attr=False)
        self.bn=nn.BatchNorm2D(out_channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return self.relu(x)
        
class ACM(nn.Layer):
    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super(ACM, self).__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.global_info = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.gla = nn.Conv2D(self.channels, self.pool_scale**2, 1, 1, 0)

        self.residual_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        # print('input acm shape ',x.shape) #[2, 2048, 64, 128]
        # print('self.pool_scale',self.pool_scale) #1
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        # print('pooled1_x',pooled_x.shape)  #[2, 2048, 1, 1]
        # [batch_size, channels, h, w]
        x = self.input_redu_conv(x)
        # print('xxxx',x.shape) #[2, 51, 64, 128]
        # [batch_size, channels, pool_scale, pool_scale]
        pooled_x = self.pooled_redu_conv(pooled_x)
        # print('pooled_x pooled_redu_conv,',pooled_x.shape) #2, 51, 1, 1]
        
        batch_size = x.shape[0]
        # [batch_size, pool_scale * pool_scale, channels]
        
        pooled_x=pooled_x.reshape((batch_size,self.channels,-1))
        pooled_x=pooled_x.transpose((0, 2, 1))
        # pooled_x = pooled_x.view(batch_size, self.channels,
        #                          -1).permute(0, 2, 1).contiguous()
        # [batch_size, h * w, pool_scale * pool_scale]
        # print('----')
        # print('pooled_x',pooled_x.shape)  #[2, 1, 512]
        tmp=self.global_info(F.adaptive_avg_pool2d(x, 1))
        
        # print('tmp.shape',tmp.shape) #2, 512, 1, 1]
        # print(x.shape)
    
        # print(x.shape[2:]) #[64, 128] 
        tmp=resize(tmp,x.shape[2:])
        # print('reszie',tmp.shape) #[2, 512, 64, 128]
        tmp=x + tmp
        # print('x+tmp',tmp.shape) #[2, 512, 64, 128]
        
        tmp=self.gla(tmp)
        # print('gla',tmp.shape) #[2, 1, 64, 128]
        
        affinity_matrix = tmp.transpose([0, 2, 3, 1]).reshape(
                                       [batch_size, -1, self.pool_scale**2])
        affinity_matrix = F.sigmoid(affinity_matrix)
        # [batch_size, h * w, channels]
        z_out = paddle.matmul(affinity_matrix, pooled_x)
        # [batch_size, channels, h * w]
        z_out = z_out.transpose([0, 2, 1])
        # [batch_size, channels, h, w]
        z_out = z_out.reshape([batch_size, self.channels, x.shape[2], x.shape[3]])
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)

        return z_out

class APCHead(nn.Layer):
    
    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, dropout_ratio=0.1):
        super(APCHead, self).__init__()
        assert isinstance(pool_scales, (list, tuple))

        self.pool_scales = pool_scales
        self.fusion = fusion
        self.in_channels=2048
        self.channels=512
        self.conv_cfg=None
        self.norm_cfg='BN'
        self.act_cfg='relu'
        self.conv_seg=nn.Conv2D(512, 19, kernel_size=1,stride=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)
        else:
            self.dropout = None
        
        acm_modules = []
        for pool_scale in self.pool_scales:
            acm_modules.append(
                ACM(pool_scale,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        # self.acm_modules = paddle.nn.Sequential(acm_modules)
        self.acm_modules = nn.LayerList(acm_modules)
        
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
    def _transform_inputs(self,x):
        return [x for i in range(5)]
    def forward(self, x):
        
        
        # print('x.shape',x.shape) #[2, 2048, 64, 128]
        acm_outs = [x]
        
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(x))
            
        # i=0
        # for ele in acm_outs:
        #     print('acm module{}'.format(i), ele.shape)
        #     i+=1
        
        acm_outs = paddle.concat(acm_outs, axis=1)
        output = self.bottleneck(acm_outs)
        output = self.cls_seg(output)
        # print('acm head out ',output.shape) #[2, 19, 64, 128]
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
    model=APCHead()
    x=paddle.normal(shape=[2, 2048, 64, 128])
    out=model(x)
    # print(model)
    # paddle.summary(model, [2, 2048, 64, 128]) #[2, 2048, 64, 128]
    