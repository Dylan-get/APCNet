import paddle
import random
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv
from paddleseg.transforms import Compose,Resize,Normalize,RandomHorizontalFlip,RandomDistort,ResizeStepScaling,RandomPaddingCrop
from paddle.fluid.dataloader.collate import default_collate_fn
from paddle.vision.transforms import CenterCrop
from pdb import set_trace as breakpoint
from paddle.io import DataLoader
import glob
import matplotlib.pyplot as plt
from paddle.io import Dataset,DataLoader
import numpy as np
from tqdm import tqdm
from  tmp import *
from config import *
class CityScapesDataset(Dataset):
    def __init__(self,root,Img_size=(1024,2048,3),Buffer_size=500,mode='train',SEED=1):
        random.seed(SEED)
        super(CityScapesDataset, self).__init__()
        if not os.path.isabs(root):
            # print(os.getcwd())
            self.root=os.path.join(os.getcwd(),root)
        else:
            self.root=root
        self.Img_size = Img_size
        self.Buffer_size = Buffer_size
        self.mode = mode
        self.imgFiles_list = sorted(glob.glob("{root}/leftImg8bit/{mode}/*/*_leftImg8bit.png".format(root=self.root,mode=self.mode)))
        self.gtFiles_list = sorted(glob.glob("{root}/gtFine/{mode}/*/*_gtFine_labelTrainIds.png".format(root=self.root,mode=self.mode)))
        # print(len(self.imgFiles_list))
        # print(len(self.gtFiles_list))
        # print(self.root)
        assert len(self.imgFiles_list)==(len(self.gtFiles_list)),'img files_list does not  match gt files'
        self.size = len(self.imgFiles_list)
        _=list(zip(self.imgFiles_list,self.gtFiles_list))
        index=np.arange(len(self.imgFiles_list))
        random.shuffle(index)
        self.data=[_[i] for i in index]
        
        transforms=[]
        if mode=='train':
            transforms.append(ResizeStepScaling(min_scale_factor=0.5,max_scale_factor=2.0,scale_step_size=0))
            transforms.append(RandomPaddingCrop((1024,512)))
            transforms.append(RandomHorizontalFlip(0.5))
            transforms.append(RandomDistort(brightness_range=0.25,brightness_prob=1,\
                                contrast_range= 0.25 \
                                ,contrast_prob=1 \
                                ,saturation_range=0.25 \
                                ,saturation_prob=1 \
                                ,hue_range=63 \
                                ,hue_prob=1))
            # transforms.append(Pad())
            transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225],
                                #   data_format='HWC'
                                  ))
            # self.transform.append(Resize(size=32))
        elif mode=='val':
            transforms.append(RandomPaddingCrop((1024,512)))
            transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225],
                                #   data_format='HWC'
                                  ))
        # elif mode=='val':
        #     transforms.append(Resize((1024,512),interp='LINEAR'))
        #     transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
        #                           std=[0.229, 0.224, 0.225],
        #                         #   data_format='HWC'
        #                           ))
        
        elif mode=='test':
            transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225],
                                #   data_format='HWC'
                                  ))
        self.transforms=Compose(transforms)
        '''
        ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 
        'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 
        'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 
        'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 
        'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 
        'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']
        '''
    def __getitem__(self, index):
        imgFile, labelFile = self.data[index]
        # print(imgFile)
        id1=imgFile.split('_')[2]
        id2=labelFile.split('_')[2]
        assert(id1==id2)
        img=np.array(Image.open(imgFile)).astype(np.uint8)#RGB hwc (1024, 2048, 3)
        label=np.array(Image.open(labelFile)).astype(np.int64)
        # img=paddle.to_tensor(img,dtype='float32')
        # label=paddle.to_tensor(img,dtype='int64')
        
        '''
        Labels:
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 255}
        '''
        # print(img.dtype)
        # img=img.transpose([2,0,1])
        if self.mode == 'test':
            im, _ = self.transforms(im=img)
            return im
        elif self.mode == 'val':
            im, label = self.transforms(im=img, label=label)
            # label = label
            return im, label
        else:
            # print(img.shape)
            im, label = self.transforms(im=img, label=label)
            # print(im.shape)
            return im, label

        
        # print(img.shape)
        # print(label.shape)
        # # img = self.transform(img)
        # # label = label[np.newaxis, :, :]
        # return img, label

    def __len__(self):
        return self.size

def main():
    print('数据处理方法：', paddle.vision.transforms.__all__)
    trainset=CityScapesDataset(root=CONFIG['dataset']['root'],mode='train',SEED=1)
    valset=CityScapesDataset(root=CONFIG['dataset']['root'],mode='val',SEED=1)
    testset=CityScapesDataset(root=CONFIG['dataset']['root'],mode='test',SEED=1)
    trainLoader=DataLoader(trainset,batch_size=CONFIG['train_batch_size'], shuffle=True)
    valLoader=DataLoader(valset,batch_size=CONFIG['val_batch_size'], shuffle=True)
    testLoader=DataLoader(testset,batch_size=CONFIG['test_batch_size'], shuffle=True)
    assert(len(trainset)+len(valset)+len(testset)==5000)
    print(len(trainset)+len(valset)+len(testset))
    labels=set()
    for batch_id, ele in enumerate(tqdm(valLoader)):
        x,y=ele
        print(x.shape)
        print(y.dtype)
        # y=y.numpy()
        # flatten=y.flatten().tolist()
        # _=set(flatten)
        # labels = labels | _
        # print(len(labels))
    print(labels)
    #(1024, 2048, 3) #h,w,c
    #plt.imshow(img)
    
    #plt.savefig('./tmp/tmp.png')
def test():
    # transforms=[]
    # transforms.append(ResizeStepScaling(min_scale_factor=0.5,max_scale_factor=2.0,scale_step_size=0))
    # transforms.append(RandomPaddingCrop((512,1024)))
    # transforms.append(RandomHorizontalFlip(0.5))
    # transforms.append(RandomDistort(brightness_range=0.25,brightness_prob=1,\
    #                             contrast_range= 0.25 \
    #                             ,contrast_prob=1 \
    #                             ,saturation_range=0.25 \
    #                             ,saturation_prob=1 \
    #                             ,hue_range=63 \
    #                             ,hue_prob=1))
    #         # transforms.append(Pad())
    # transforms.append(Normalize(mean=[0.485, 0.456, 0.406], 
    #                               std=[0.229, 0.224, 0.225],
    #                             #   data_format='HWC'
    #                               ))
    img=Image.open('/app/wht/paddle/apcnet/dataset/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png')
    img=np.array(img).astype('uint8') #1024 2048 3
    crop=Compose([RandomPaddingCrop((1024,512))]) #w,h
    # crop=RandomPaddingCrop([1024,512])
    print(img.shape)
    plt.imshow(img)
    plt.savefig('./tmp/crop1.png')
    img=crop(img)   #compose (3, 512, 1024)  no compose:(512, 1024, 3)
    print(img[0].shape)
    _=img[0].transpose([1,2,0]) #(512, 1024, 3)
    print(_.shape)
    plt.imshow(_)
    plt.savefig('./tmp/crop2.png')
if __name__=="__main__":
    # test()
    main()