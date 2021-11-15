from tqdm import tqdm
import paddle
import datetime
import logging
import os
from dataloader import CityScapesDataset
from architectures import getApcNet
from paddle.io import Dataset, DataLoader,BatchSampler,DistributedBatchSampler
from config import CONFIG
import paddle.nn.functional as F
import warnings
import glob
import numpy as np
import math
import paddle.nn as nn
import paddle.distributed as dist
import time
# warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, expName, resume=True, resume_inter='latest', config=None):
        super(Trainer, self).__init__()
        print('---init---')
        self.max_inter = int(80000 * 8 / CONFIG['train_batch_size'])
        self.inter = 1
        self.optimizers_step_size = 1  # int(self.max_inter / (math.log(0.01,0.9))) #320000/43,44
        # print('self.optimizers_step_size', self.optimizers_step_size)
        self.config = CONFIG
        self.root = os.getcwd()
        self.resume = resume
        self.resume_inter = resume_inter
        self.best_miou = -1.0
        self.best_miou_inter = -100
        print('---init exp_dir---')
        self.exp_dir = self.init_exp_dir(expName=expName)
        print('---init logger---')
        self.logger = self.init_logger()
        print('---init models---')
        self.models = self.init_models()
        print('---init dataloaders---')
        self.dataloaders = self.init_dataloaders()
        self.optimizers = self.init_optimizers()
        self.criterions = self.init_criterions()
        self.init_check()
        self.stepEachEpoch = (len(self.dataloaders['train']) + 1)
        self.train_display_step = 50
        self.train_save_step = 50
        self.val_step = 2000
        self.logger.info('init complete')


    def run(self):
        ##resume
        if self.resume:
            self.resume_experiment()
        self.Loss = {}
        self.Loss['step_cnt'] = 0
        self.Loss['apc_loss'] = 0.0
        self.Loss['fcn_loss'] = 0.0
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        while (self.inter < self.max_inter):
            for batch_id, batch in enumerate(tqdm(self.dataloaders['train']())):
                t0=time.perf_counter()
                if self.inter % self.optimizers_step_size == 0:
                    self.change_learing_rate()
                # t1=time.perf_counter()
                lossapc, lossfcn = self.train_step(batch_id, batch)
                # t2=time.perf_counter()
                self.Loss['step_cnt'] += 1
                self.Loss['apc_loss'] += lossapc
                self.Loss['fcn_loss'] += lossfcn
                # t2=time.perf_counter()
                if self.inter % self.train_display_step == 0:
                    self.logger.info('inter:{} , APC Loss:{:.5f} FCN Loss:{:.5f},lr:{:.5f}' \
                                     .format(self.inter,
                                             float(self.Loss['apc_loss'] / self.Loss['step_cnt']),
                                             float(self.Loss['fcn_loss'] / self.Loss['step_cnt']),
                                             float(self.optimizers['APCHead'].get_lr()),
                                             )
                                     )
                    self.Loss['step_cnt'] = 0
                    self.Loss['apc_loss'] = 0.0
                    self.Loss['fcn_loss'] = 0.0
                if self.inter % self.train_save_step == 0:
                    self.save_checkpoint()
                    self.delete_checkpoint()

                if self.inter % self.val_step == 0:
                    miou, ious = self.val()
                    if miou > self.best_miou:
                        self.best_miou = miou
                        self.best_miou_inter = self.inter
                        self.save_checkpoint(postfix='best')
                self.inter += 1
                # print(f'spend 1:{t1-t0},spend 2:{t2-t1}')
    def resume_experiment(self):
        self.load_checkpoint()
        # print('self.inter',self.inter)
        # for i in range(self.inter-1):
        #     print(i)
        #     # self.schedulers['backbone'].step()
        #     self.schedulers['APCHead'].step()
        #     if i== 50:
        #         break
        # self.schedulers['FCNHead'].step()
        # self.scheduler['backbone'].step(self.inter-1)
        # self.scheduler['APCHead'].step(self.inter-1)
        # self.scheduler['FCNHead'].step(self.inter-1)

    def change_learing_rate(self):

        s1 = self.optimizers['backbone'].get_lr()
        s2 = self.optimizers['APCHead'].get_lr()
        s3 = self.optimizers['FCNHead'].get_lr()

        z = math.log(0.01, 0.9)
        decay = 99.0 / self.max_inter
        # 0.01=0.9^z
        # lr_backbone = math.pow(0.9, z*self.inter/ self.max_inter) * self.learning_rate_start['backbone']
        # lr_apchead = math.pow(0.9, z*self.inter/ self.max_inter) * self.learning_rate_start['APCHead']
        # lr_fcnhead = math.pow(0.9, z*self.inter/ self.max_inter) * self.learning_rate_start['FCNHead']
        lr_backbone = self.learning_rate_start['backbone'] / (1 + decay * self.inter)
        lr_apchead = self.learning_rate_start['APCHead'] / (1 + decay * self.inter)
        lr_fcnhead = self.learning_rate_start['FCNHead'] / (1 + decay * self.inter)

        # print((lr_apchead))\
        ##wram up
        if self.inter < 2000:
            lr_backbone =(0.1+0.9*(self.inter/2000))*self.learning_rate_start['backbone']
            lr_apchead = self.learning_rate_start['APCHead']/10
            lr_fcnhead = self.learning_rate_start['FCNHead']/10
        self.optimizers['backbone'].set_lr(lr_backbone)
        self.optimizers['APCHead'].set_lr(lr_apchead)
        self.optimizers['FCNHead'].set_lr(lr_fcnhead)

        n1 = self.optimizers['backbone'].get_lr()
        n2 = self.optimizers['APCHead'].get_lr()
        n3 = self.optimizers['FCNHead'].get_lr()
        # self.logger.info(
        #     'adjust resNet lr from {} to {}, APCHead lr from {} to {},FCNHead lr from {} to {}'.format\
        #         (
        #         s1,n1,s2,n2,s3,n3
        #     ))

    def init_exp_dir(self, expName):
        expdir = os.path.join(self.root, 'experiments', expName)
        if os.path.exists(expdir) and not self.resume:
            # now_str = datetime.datetime.now().__str__().replace(' ','_')
            # expdir=os.path.join(self.root,'experiments',expName+'_'+now_str)
            # if os.path.exists(expdir):
            print('exist exp dir {}'.format(expdir))
            exit(1)
            # else:
            #     os.makedirs(expdir)
        else:

            os.makedirs(expdir, exist_ok=True)
            models_dir = os.path.join(expdir, 'ckpt')
            os.makedirs(models_dir, exist_ok=True)
        return expdir

    def init_logger(self):
        logger = logging.getLogger(__name__)
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        logger.addHandler(strHandler)
        logger.setLevel(logging.INFO)
        log_dir = os.path.join(self.exp_dir, 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().__str__().replace(' ', '_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_' + now_str + '.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        logger.addHandler(self.log_fileHandler)
        return logger

    def init_models(self):
        print('----')
        networks, msg_resnet = getApcNet(config=self.config)
        print('----apcnet')
        pre_apcent = 0
        if pre_apcent:
            state = paddle.load(
                './architectures/pretrained/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pdparams')
            networks['backbone'].set_state_dict(state['models']['backbone'])
            networks['APCHead'].set_state_dict(state['models']['APCHead'])
            networks['FCNHead'].set_state_dict(state['models']['FCNHead'])
        for param in networks['backbone'].parameters():
            param.requires_grad = False
        networks['backbone'].eval()
        print('----log')
        self.logger.info(msg_resnet)
        print('----')
        networks['backbone']=paddle.DataParallel(networks['backbone'])
        networks['APCHead']=paddle.DataParallel(networks['APCHead'])
        networks['FCNHead']=paddle.DataParallel(networks['FCNHead'])
        return networks  ##['backbone','APCHead','FCNHead']
        pass

    def init_dataloaders(self):
        # trainLoader,valLoader,testLoader
        trainset = CityScapesDataset(root=self.config['dataset']['root'], mode='train', SEED=1)
        valset = CityScapesDataset(root=self.config['dataset']['root'], mode='val', SEED=1)
        testset = CityScapesDataset(root=self.config['dataset']['root'], mode='test', SEED=1)

        trainset_sample=DistributedBatchSampler(dataset=trainset,
                                        batch_size=self.config['train_batch_size'],
                                        shuffle=True, drop_last=False)

        valset_sample=DistributedBatchSampler(dataset=valset,
                                        batch_size=self.config['val_batch_size'],
                                        shuffle=False, drop_last=False)

        testset_sample=DistributedBatchSampler(dataset=testset,
                                        batch_size=self.config['test_batch_size'],
                                        shuffle=False, drop_last=False)

        trainLoader = DataLoader(trainset, batch_sampler=trainset_sample,
                                 num_workers=self.config['num_workers'],use_buffer_reader=True)
        valLoader = DataLoader(valset, batch_sampler=valset_sample,
                               num_workers=self.config['num_workers'], use_buffer_reader=True)
        testLoader = DataLoader(testset,batch_sampler=testset_sample,
                                num_workers=self.config['num_workers'], use_buffer_reader=True)
        print(len(trainset))
        print(len(valset))
        print(len(testset))

        assert len(trainset) + len(valset) + len(testset) == 5000, 'images number is not 5000'
        assert len(trainset) == 2975
        assert len(valset) == 500
        assert len(testset) == 1525
        dataloaders = {}
        dataloaders['train'] = trainLoader
        dataloaders['val'] = valLoader
        dataloaders['test'] = testLoader
        return dataloaders

    def init_optimizers(self):

        optimizers = {}
        # self.schedulers = {}
        backboneCfg = self.config['optimizers']['backbone']
        apcnetCfg = self.config['optimizers']['APCHead']
        fcnheadCfg = self.config['optimizers']['FCNHead']
        self.learning_rate_start = dict(backbone=backboneCfg['lr'], APCHead=apcnetCfg['lr'], FCNHead=apcnetCfg['lr'])

        optimizers['backbone'] = paddle.optimizer.Momentum(
            parameters=self.models['backbone'].parameters(),
            learning_rate=self.learning_rate_start['backbone'],
            momentum=backboneCfg['momentum'],
            weight_decay=backboneCfg['weight_decay'])
                                                             verbose=True)
        optimizers['APCHead'] = paddle.optimizer.Momentum(
            parameters=self.models['APCHead'].parameters(),
            learning_rate=self.learning_rate_start['APCHead'],
            momentum=apcnetCfg['momentum'],
            weight_decay=apcnetCfg['weight_decay'])
            
        optimizers['FCNHead'] = paddle.optimizer.Momentum(
            parameters=self.models['FCNHead'].parameters(),
            learning_rate=self.learning_rate_start['FCNHead'],
            momentum=fcnheadCfg['momentum'],
            weight_decay=fcnheadCfg['weight_decay'])
        return optimizers

    def init_criterions(self):
        criterions = {}
        criterions['celoss'] = paddle.nn.CrossEntropyLoss(ignore_index=255, reduction='mean', axis=1)
        return criterions
        pass

    def init_check(self):
        assert len(self.models) == 3
        assert len(self.optimizers) == 3
        assert len(self.criterions) == 1

    def train_step(self, batch_id, batch):
        
        
        self.models['backbone'].train()
        self.models['APCHead'].train()
        self.models['FCNHead'].train()
        t1=time.perf_counter()
        self.optimizers['backbone'].clear_grad()
        self.optimizers['APCHead'].clear_grad()
        self.optimizers['FCNHead'].clear_grad()
        t2=time.perf_counter()

        x, label = batch
        # print(x.shape)
        x = paddle.to_tensor(x, dtype='float32')
        label = paddle.to_tensor(label, dtype='int64')
        # print('label',label.shape)
        # t2=time.perf_counter()
        # with paddle.amp.auto_cast():
        feature2, feature3 = self.models['backbone'](x)  # 0,1,2,3 fcnhead

        pre1 = self.models['APCHead'](feature3)
        pre1 = F.interpolate(x=pre1, size=[512, 1024], mode="bilinear", align_corners=False)
        loss_apc = 1.0 * self.criterions['celoss'](pre1, label)
        t3=time.perf_counter()

        loss_apc_numpy = loss_apc.numpy()

        pre2 = self.models['FCNHead'](feature2)
        pre2 = F.interpolate(x=pre2, size=[512, 1024], mode="bilinear", align_corners=False)
        loss_fcn = 0.4 * self.criterions['celoss'](pre2, label)
        # loss_fcn.backward()
        loss_fcn_numpy = loss_fcn.numpy()
        t4=time.perf_counter()

        loss = loss_apc +0.4*loss_fcn
        # scaled = self.scaler.scale(loss)
        # scaled.backward()

        loss.backward()
        t5=time.perf_counter()

        # self.scaler.minimize(self.optimizers['backbone'],scaled)
        # self.scaler.minimize(self.optimizers['APCHead'],scaled)
        # self.scaler.minimize(self.optimizers['FCNHead'],scaled)
        self.optimizers['backbone'].step()
        self.optimizers['APCHead'].step()
        self.optimizers['FCNHead'].step()

        t6=time.perf_counter()

        print(f'spend time:{t2-t1} {t3-t2} {t4-t3} {t5-t4} {t6-t5} ')
        return loss_apc_numpy, loss_fcn_numpy  # (apc,fcn)

    def val(self):
        self.models['backbone'].eval()
        self.models['APCHead'].eval()
        self.models['FCNHead'].eval()
        ignore_label255 = 255  # 0 ,255
        intersection,union=0,0
        for i, batch in enumerate(tqdm(self.dataloaders['val'])):
            x, label = batch
            x = paddle.to_tensor(x, dtype='float32')
            # print(label.shape)
            label = paddle.to_tensor(label, dtype='int64')
            # print('label',label.shape)
            feature2, feature3 = self.models['backbone'](x)  # 0,1,2,3
            # print('feature.shape',feature.shape)
            pre1 = self.models['APCHead'](feature3)
            pre2 = self.models['FCNHead'](feature2)
            pre1 = 1.0 * pre1 + 0.4 * pre2
            pre1 = F.interpolate(x=pre1, size=[512, 1024], mode="bilinear")  # ,align_corners=True
            # pre2=F.interpolate(x=pre2, size=[512,1024])
            prediction = paddle.argmax(pre1, axis=1).numpy()

            miou_tmp, intersection_tmp, union_tmp = self.getMiou(prediction, label.numpy(), ignore_label=ignore_label255)
            intersection+=intersection_tmp
            union+=union_tmp
            del x, label, feature2, feature3, pre1, pre2, prediction
        ious=intersection/union
        miou=np.mean(ious)
        self.logger.info('inter {} ,val miou:{}'.format(self.inter, miou))
        return miou, ious

    def save_checkpoint(self, postfix='normal'):
        state = {}
        state['inter'] = self.inter
        state['best'] = {}
        state['best']['best_miou'] = self.best_miou
        state['best']['best_miou_inter'] = self.best_miou_inter
        state['models'] = {}
        state['models']['backbone'] = self.models['backbone'].state_dict()
        state['models']['APCHead'] = self.models['APCHead'].state_dict()
        state['models']['FCNHead'] = self.models['FCNHead'].state_dict()
        state['optimizers'] = {}
        state['optimizers']['APCHead'] = self.optimizers['APCHead'].state_dict()
        state['optimizers']['FCNHead'] = self.optimizers['FCNHead'].state_dict()

        if postfix == 'best':
            save_path = os.path.join(self.exp_dir, 'ckpt',
                                     str(self.inter) + '.pdparams.bestmIou_{}'.format(self.best_miou))
            paddle.save(state, save_path)
            self.logger.info('save best ckpt inter:{}, mIou : {}'.format(self.inter, self.best_miou))
        else:
            save_path = os.path.join(self.exp_dir, 'ckpt', str(self.inter) + '.pdparams')
            paddle.save(state, save_path)
            self.logger.info('save ckpt inter:{}'.format(self.inter))

    def delete_checkpoint(self):
        ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
        for ckpt in ckpts:
            inter = int(ckpt.split('/')[-1].split('.')[0])
            if inter != self.inter:
                delete_path = os.path.join(self.exp_dir, 'ckpt', str(inter) + '.pdparams')
                os.remove(delete_path)

    def load_checkpoint(self):
        if self.resume_inter == 'latest':

            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            pass
        else:
            load_path = os.path.join(self.exp_dir, 'ckpt', str(self.resume_inter) + '.pdparams')
        # print(load_path)
        try:
            state = paddle.load(load_path)
        except:
            if latest == -1:
                self.logger.info("no ckpt, no load ckpt")
                return
            os.remove(load_path)
            ckpts = list(glob.glob(self.exp_dir + '/ckpt/*.pdparams'))
            latest = -1
            for ckpt in ckpts:
                inter = int(ckpt.split('/')[-1].split('.')[0])
                if inter > latest:
                    latest = inter
            load_path = os.path.join(self.exp_dir, 'ckpt', str(latest) + '.pdparams')
            state = paddle.load(load_path)
        self.inter = state['inter'] + 1
        self.best_miou = state['best']['best_miou']
        self.best_miou_inter = state['best']['best_miou_inter']
        self.models['backbone'].set_state_dict(state['models']['backbone'])
        self.models['APCHead'].set_state_dict(state['models']['APCHead'])
        self.models['FCNHead'].set_state_dict(state['models']['FCNHead'])
        self.optimizers['APCHead'].set_state_dict(state['optimizers']['APCHead'])
        self.optimizers['FCNHead'].set_state_dict(state['optimizers']['FCNHead'])
        self.logger.info('resume ckpt from {}'.format(load_path))

    def getMiou(self,prediction, target, ignore_label=255):
        confusionMatrix = np.zeros((19, 19), dtype=int)
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        for i in range(prediction.shape[0]):
            if target[i] != ignore_label:
                confusionMatrix[target[i], prediction[i]] += 1

        intersection = np.diag(confusionMatrix)
        ground_truth_set = confusionMatrix.sum(axis=1)
        predicted_set = confusionMatrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        return np.mean(IoU), intersection, union
if __name__ == '__main__':
    # p=dict(a=1,b=2)
    # print(len(p))
    # print(len(p.items()))
    # local_master = (args.local_rank == -1 or dist.get_rank() == 0)
    local_master = dist.get_rank() == 0
    dist.init_parallel_env()
    trainer = Trainer(expName='apcnet-cityscapes-freeze-warmup-decay99-bn', resume=1, resume_inter='latest',
                      config=CONFIG)
    ##freeze
    ##no freeze
    ##decay 99 backbone lr0.01
    ##decay 9  backbone lr0.001

    trainer.run()
