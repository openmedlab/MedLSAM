#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
from ast import While
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import argparse
from torchvision import transforms
from train.dataloaders.Position_dataloader import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.parse_config import parse_config
from networks.NetFactory import NetFactory
from losses.EmbFCLoss import *
from util.load_save_model import Save_checkpoint
from util.logger import Logger

def random_all(random_seed):
    random.seed(random_seed) 
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def train(config_file):
    # 1, load configuration parameters
    writer = SummaryWriter()
    logger = Logger()
    logger.print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']    # data config, e.g. data_shape,batch_size
    config_net  = config['network'] # net config, e.g. net_type,base_feature_name,class_num
    config_train = config['training']
    patch_size = config_data['patch_size']
    batch_size = config_data.get('batch_size', 4)
    dis_ratio = torch.tensor(config_data.get('distance_ratio'), dtype=torch.float32).unsqueeze(dim=0)
    crop_pad = 8 # crop
    logger.print('dis ratio', dis_ratio)
    cur_loss = 0
    lr = config_train.get('learning_rate', 1e-3)
    best_loss = config_train.get('best_loss', 0.5)
    random_seed = config_train.get('random_seed', 1)
    random_all(random_seed)    
    save_model = Save_checkpoint()
    
    
    cudnn.benchmark = True
    cudnn.deterministic = True

    # 2, load data
    logger.print('2.Load data')
    trainData = PositionDataloader(iter_num=config_data['iter_num'],
                                   image_list=config_data['train_image_list'],
                                   transform=transforms.Compose([
                                       RandomDoubleCrop(patch_size, small_move=False, fluct_range=[60,60,60], crop_pad=crop_pad),
                                       RandomDoubleIntensity(k=0.1, hu_ls=[0.2, 0.4, 0.6, 0.8], prob=0.5),
                                        RandomDoubleNoise(mean=0, std=0.2, prob=0.5),
                                        RandomDoubleAffine(scales=[0.1,0.1,0.1], degrees=[15,15,15], prob=0.5),
                                        RandomDoubleElasticDeformation(num_control_points=[5,5,5], max_displacement=[3,3,3],prob=0.5),
                                       ToPositionTensor(crop_pad=crop_pad, output_size=patch_size),
                                   ]),
                                   load_memory=False, 
                                   parral_load=False,        
                                   random_sample=True,
                                   out_size=patch_size,
                                   crop_pad=crop_pad,
                                   batch_size=batch_size)
    
    validData = PositionDataloader(iter_num=50,
                                   image_list=config_data['valid_image_list'],
                                   transform=transforms.Compose([
                                       RandomDoubleCrop(patch_size, small_move=False, fluct_range=[60,60,60]),
                                       ToPositionTensor(output_size=patch_size)
                                   ]),
                                   load_memory=False,
                                   parral_load=False,     
                                   random_sample=True,
                                   out_size=patch_size)
    trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=False)
    validLoader = DataLoader(validData, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # 3. creat model
    logger.print('3.Creat model')
    
    net_type   = config_net['net_type']
    net_class = NetFactory.create(net_type)
    patch_size = np.asarray(config_data['patch_size'])
    net = net_class(
                    inc=config_net.get('input_channel', 1),
                    patch_size=patch_size,
                    base_chns= config_net.get('base_feature_number', 16),
                    norm='in',
                    depth=config_net.get('depth', False),
                    dilation=config_net.get('dilation', 1),
                    n_classes = config_net['class_num'],
                    droprate=config_net.get('drop_rate', 0.2),
                    )
    net = torch.nn.DataParallel(net).cuda()
    rdrloss = nn.MSELoss()
    CalLoss = EmbFCLoss(dis_ratio, rdrloss, embratio=10, crop_pad=crop_pad, dis_thresh=3)
    CalLoss=torch.nn.DataParallel(CalLoss).cuda()
    Adamoptimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.StepLR(Adamoptimizer, step_size=10, gamma=0.8)
    if config_train['load_pretrained_model']:
        logger.print("=> loading checkpoint '{}'".format(config_train['pretrained_model_path']))
        if config_train['pretrained_model_path'].endswith('.tar'):
            checkpoint = torch.load(config_train['pretrained_model_path'])
            net.load_state_dict(checkpoint['state_dict'])
            Adamoptimizer.load_state_dict(checkpoint['optimizer'])
        elif config_train['pretrained_model_path'].endswith('.pkl'):
            net_weight = torch.load(config_train['pretrained_model_path'], map_location=lambda storage, loc: storage)
            net.load_state_dict(net_weight)
        logger.print("=> loaded checkpoint '{}' ".format(config_train['pretrained_model_path']))
    Adamoptimizer.lr = lr


    # 4, start to train
    logger.print('4.Start to train')
    start_it  = config_train.get('start_iteration', 0)
    print_iter = 0
    for epoch in range(start_it, config_train['maximal_epoch']):  
        logger.print('#######epoch:', epoch)
        optimizer = Adamoptimizer
        net.train()
        for i_batch, sample_batch in enumerate(trainLoader):
            img_batch0, img_batch1, fc_label_batch, \
            fs_po0, fs_po1 = sample_batch['random_crop_image_0'].cuda(), \
                sample_batch['random_crop_image_1'].cuda(), \
                sample_batch['rela_distance'].cuda(),\
                sample_batch['random_fullsize_position_0'].cuda(), \
                sample_batch['random_fullsize_position_1'].cuda()
            ori_img_batch0, ori_img_batch1, ori_fc_label_batch, \
            ori_fs_po0, ori_fs_po1 = sample_batch['ori_random_crop_image_0'].cuda(), \
                sample_batch['ori_random_crop_image_1'].cuda(), \
                sample_batch['ori_rela_distance'].cuda(),\
                sample_batch['ori_random_fullsize_position_0'].cuda(), \
                sample_batch['ori_random_fullsize_position_1'].cuda()
            predic_0 = net(x=img_batch0, out_fc=True, decoder=True, out_feature=True)
            predic_1 = net(x=img_batch1, out_fc=True, decoder=True, out_feature=True)
            ori_predic_0 = net(ori_img_batch0, out_fc=True, decoder=True, out_feature=True)
            ori_predic_1 = net(ori_img_batch1, out_fc=True, decoder=True, out_feature=True)
            [train_loss, fc_train_loss, emb_loss, fc_predic]= \
                CalLoss(ori_img_batch0, ori_img_batch1, fs_po0, fs_po1, ori_fs_po0, ori_fs_po1, fc_label_batch, \
                            ori_fc_label_batch, predic_0, predic_1, ori_predic_0, ori_predic_1)
            optimizer.zero_grad() 
            train_loss.sum().backward() 
            optimizer.step() 
            if i_batch%config_train['print_step']==0:
                fc_train_loss = fc_train_loss.sum().cpu().data.numpy()
                emb_loss = emb_loss.sum().cpu().data.numpy()
                train_loss = train_loss.sum().cpu().data.numpy()
                fc_predic = fc_predic[0].cpu().data.numpy()
                ori_fc_label_batch = ori_fc_label_batch[0].cpu().data.numpy()
                writer.add_scalar('Loss/train_fc', fc_train_loss, print_iter)
                writer.add_scalar('Loss/train_emb', emb_loss, print_iter)
                print_iter+=1
                if i_batch ==0:
                    train_loss_array=fc_train_loss
                else:
                    train_loss_array = np.append(train_loss_array, fc_train_loss)
                logger.print('train batch:',i_batch,'rdr:', fc_train_loss,'emb:', emb_loss, \
                        'label:', ori_fc_label_batch, 'predic', fc_predic)
                logger.flush()
        Adamscheduler.step()
        
        with torch.no_grad():
            net.eval()
            for ii_batch, sample_batch in enumerate(validLoader):
                img_batch0, img_batch1, fc_label_batch, \
                fs_po0, fs_po1 = sample_batch['random_crop_image_0'].cuda(), \
                    sample_batch['random_crop_image_1'].cuda(), \
                    sample_batch['rela_distance'].cuda(),\
                    sample_batch['random_fullsize_position_0'].cuda(), \
                    sample_batch['random_fullsize_position_1'].cuda()                   
                predic_0 = net(x=img_batch0, out_fc=True)
                predic_1 = net(x=img_batch1, out_fc=True)
                predic_cor_fc_0  = predic_0['fc_position']
                predic_cor_fc_1 = predic_1['fc_position']
                fc_predic = dis_ratio.cuda()*torch.tanh(predic_cor_fc_0-predic_cor_fc_1)
                fc_valid_loss = rdrloss(fc_predic, fc_label_batch).cpu().data.numpy()
                
                if ii_batch ==0:
                    valid_loss_array = fc_valid_loss
                else:
                    valid_loss_array = np.append(valid_loss_array, fc_valid_loss)
                logger.print('valid batch:',ii_batch,' valid loss:', fc_valid_loss)
                logger.flush()

            epoch_loss = {'valid_loss':valid_loss_array.mean(), 'train_loss':train_loss_array.mean()}
            cur_loss = valid_loss_array.mean()
            writer.add_scalar('Loss/valid_fc', cur_loss, epoch)
            t = time.strftime('%X %x %Z')
            logger.print(t, 'epoch', epoch, '\nloss:\n', epoch_loss)

            'save current model'
            filename = config_train['model_save_name'] + "_cur_{0:}.tar".format(cur_loss)
            if cur_loss < best_loss:
                best_loss =cur_loss
                is_best = True
                bestname = config_train['model_save_name'] + "_{0:}.tar".format(cur_loss)
            else:
                is_best = False
                bestname = None
            save_model.save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=filename, bestname=bestname)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='../config/train/CMU/train_cls_111.txt')
    config_file = parser.parse_args().config
    assert(os.path.isfile(config_file))
    train(config_file)
