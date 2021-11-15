from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset,DataLoader
from  tmp import CONFIG
import paddle.nn.functional as F
import warnings
import glob 
import numpy as np
from architectures import getApcNet
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from cv2 import imread, imwrite
from PIL import Image
import shutil
def setdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
def getMiou(mat,labels_num=19):
        ious=[]
        for i in range(labels_num):
            iou_i=mat[i][i]/(np.sum(mat[i],axis=0)+np.sum(mat[:,i],axis=0)-mat[i][i])
            ious.append(iou_i)
        
        res=(np.mean(ious),ious)
        return res
        # miou=np.trace(mat)/(2*np.sum(mat)-np.trace(mat))
        # return miou,1

def get_color(mat,class_num):
    colorMaps=dict(
        label0=[220,20,60],
        label1=[139,0,139],
        label2=[106,90,205],
        label3=[0,0,205],
        label4=[65,105,225],
        label5=[135,206,250],
        label6=[225,255,255],
        label7=[0,128,128],
        label8=[127,255,170],
        label9=[46,139,87],
        label10=[50,205,50],
        label11=[0,100,0],
        label12=[255,255,0],
        label13=[255,215,0],
        label14=[255,165,0],
        label15=[160,82,45],
        label16=[255,69,0],
        label17=[250,128,114],
        label18=[255,0,0],
        label255=[255,255,255]
    )
    res=np.zeros((512,1024,3),dtype='uint8')
    for i in range(class_num):
        r=[2*i,5*i,10*i]
        res[np.where(mat==i)]=colorMaps[str('label{}'.format(i))]
    res[np.where(mat==255)]=colorMaps['label255']
    return res

def CRFs(img,prediction, save_path):

    # 计算predicted_image中的类数。
    n_labels = 19
    
    use_2d = True
   
    if use_2d:
        # 使用densecrf2d类
        # print(img.shape)
        d = dcrf.DenseCRF2D(img.shape[1],img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(prediction, n_labels, gt_prob=0.2, zero_unsure=None)
        img = np.ascontiguousarray(img)
        
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
      
    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(10)

    # 找出每个像素最可能的类
    # print(np.array(Q).shape)
    MAP = np.argmax(Q, axis=0)
    print(MAP.shape)
    prediction=MAP.reshape((512,1024))
    
    map=get_color(prediction,class_num=19)
    img=Image.fromarray(map)
    img.save(save_path)
    return prediction[np.newaxis,:,:]
def getConfusionMatrix(prediction,target,ignore_label=255):
    if ignore_label:
        confusionMatrix=np.zeros((19,19),dtype=int)
        prediction=prediction.reshape(-1)
        target=target.reshape(-1)
        for (p1,p2) in zip(target,prediction):
            if p1!=ignore_label:
                confusionMatrix[p1,p2]+=1
        return confusionMatrix
    else:
        confusionMatrix=np.zeros((20,20),dtype=int)
        prediction=prediction.reshape(-1)
        target=target.reshape(-1)
        for (p1,p2) in zip(target,prediction):
            if p1!=255:
                confusionMatrix[p1,p2]+=1
            elif p1==255:
                confusionMatrix[19,19]+=1
        return confusionMatrix
        

def test():
    valDataset=CityScapesDataset(root='../dataset/cityscapes',mode='val',SEED=1)
    valLoader=DataLoader(valDataset,batch_size=1,drop_last=False,num_workers=4, shuffle=False,use_buffer_reader=True)
    models,msg_resnet=getApcNet()
    state=paddle.load('./experiments/apcnet-cityscapes/ckpt/final.pdparams')
    models['backbone'].set_state_dict(state['models']['backbone'])
    models['APCHead'].set_state_dict(state['models']['APCHead'])
    models['FCNHead'].set_state_dict(state['models']['FCNHead'])
    
    
    ignore_label=None
    # ignore_label=255
    if ignore_label:
        confusionMatrix=np.zeros((19,19))
    else:
        confusionMatrix=np.zeros((20,20))
    setdir('./tmp/cityscape/source')
    setdir('./tmp/cityscape/pre')
    setdir('./tmp/cityscape/crfs')
    setdir('./tmp/cityscape/gt')
    for i,batch in enumerate(tqdm(valLoader())):
            x,label=batch
            
            models['backbone'].eval()
            models['APCHead'].eval()
            models['FCNHead'].eval()
           
            x=paddle.to_tensor(x,dtype='float32')
            # print(label.shape)
            label=paddle.to_tensor(label,dtype='int64')
            # print('label',label.shape)
            feature2=models['backbone'](x)[0]
            feature3=models['backbone'](x)[1] #0,1,2,3
            # print('feature.shape',feature.shape)
            pre1=models['APCHead'](feature3)
            pre2=models['FCNHead'](feature2)
            pre1=1.0*pre1+0.4*pre2
            pre1=F.interpolate(x=pre1, size=[512,1024],mode="bilinear")
            # pre2=F.interpolate(x=pre2, size=[512,1024])
            prediction=paddle.argmax(pre1,axis=1).numpy()
            
            if not ignore_label:
                prediction[np.where(label.numpy()==255)]=255
            else:
                pass
            # sourceImg=x[0].transpose([1,2,0])
            # mean=[0.485, 0.456, 0.406]
            # std=[0.229, 0.224, 0.225]
            # sourceImg=np.array([(sourceImg[:,:,i]*std[i]+mean[i])*255.0 for i in range(3)],dtype='uint8')
            # sourceImg=sourceImg.transpose([1,2,0])
            # path='./tmp/crfs/crfs/{}.png'.format(i)
            # map=CRFs(sourceImg,prediction[0],path)
            
            # print('map', map.shape)
            # print('prediction.shape',prediction.shape)
            # prediction=paddle.argmax(pre2,axis=1).numpy()
            confusionMatrix+=getConfusionMatrix(prediction,label.numpy(),ignore_label)
            # confusionMatrix+=getConfusionMatrix(map,label.numpy())
            # print(prediction[0].shape)
            
            
            
            # print(x[0].shape)
           
            save_flag=0
            if save_flag:
                sourceImg=x[0].transpose([1,2,0])
                mean=[0.485, 0.456, 0.406]
                std=[0.229, 0.224, 0.225]
                sourceImg=np.array([(sourceImg[:,:,i]*std[i]+mean[i])*255.0 for i in range(3)],dtype='uint8')
                sourceImg=sourceImg.transpose([1,2,0])
                img=Image.fromarray(sourceImg)
                img.save('./tmp/cityscape/source/{}.png'.format(i))
                
                color_pre=get_color(prediction[0],19)
                img=Image.fromarray(color_pre.astype('uint8'))
                img.save('./tmp/cityscape/pre/{}.png'.format(i))
                
                color_label=get_color(label[0],19)
                img=Image.fromarray(color_label.astype('uint8'))
                img.save('./tmp/cityscape/gt/{}.png'.format(i))
            
            
            del x,label,feature2,feature3,pre1,pre2
            # CRFs()     
    miou,_=getMiou(confusionMatrix,19 if ignore_label else 20)
    print('miou:{}'.format(miou))

def testCRFs():
    labels=np.random.ranint(0,19,(512,1024))
    d = dcrf.DenseCRF2D(512, 1024, 19)
    U = unary_from_labels(labels, 19, gt_prob=0.2, zero_unsure=None)
    d.setUnaryEnergy(U)
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
def main():
    test()
if __name__=='__main__':
    main()