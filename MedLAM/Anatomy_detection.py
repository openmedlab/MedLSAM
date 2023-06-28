#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.backends.cudnn as cudnn
from util.parse_config import parse_config
from networks.medlam import MedLAM
from data_process.data_process_func import *
from MedLAM.detection_functions import *
from MedLAM.Predic_corner_cor import predic_corner_cor
from MedLAM.Cal_support_feature import process_support
import numpy as np

random_all(2023) 

class AnatomyDetection(object):
    def __init__(self, config_file):
        # 1, load configuration parameters
        config = parse_config(config_file)
        config_data = config['data'] 
        config_weight = config['weight']

        self.dis_ratio = np.array([1500, 600, 600])[np.newaxis,:]
        self.patch_size = np.array([64, 64, 64])
        support_image_ls = read_file_list(config_data['support_image_ls'])
        support_label_ls = read_file_list(config_data['support_label_ls'])
        self.random_crop = RandomPositionCrop(self.patch_size, padding=False)
        self.fg_classes = config_data['fg_class']
        cudnn.deterministic = True

        # 2. creat model
        MedLAMNet = MedLAM(
                        inc=1,
                        patch_size = self.patch_size,
                        base_chns= 24,
                        n_classes = 3,
                        )
        MedLAMNet = torch.nn.DataParallel(MedLAMNet).half().cuda()
        if os.path.isfile(config_weight['medlam_load_path']):
            print("=> loading medlam checkpoint '{}'".format(config_weight['medlam_load_path']))
            MedLAMNet.load_state_dict(torch.load(config_weight['medlam_load_path']))
            print("=> loaded medlam checkpoint '{}' ".format(config_weight['medlam_load_path']))
        else:
            raise(ValueError("=> no checkpoint found at '{}'".format(config_weight['medlam_load_path'])))
        MedLAMNet.eval()
        self.RD = Relative_distance(MedLAMNet,out_mode='fc_position', feature_refine_mode='corner',distance_mode='tanh', \
                    center_patch_size=[8,8,8], distance_ratio=self.dis_ratio)

        # 3, start to detect
        self.iter_patch_num = 10
        with torch.no_grad():
            for fg_class in config_data['fg_class']:
                print('Loading support class: ', fg_class)
                self.RD=process_support(self.RD, support_image_ls, support_label_ls, self.patch_size, fg_class)

    def get_extreme_corner(self, query_image_path):
        with torch.no_grad():
            extreme_cor_dic, corner_cor_dic, ori_shape = predic_corner_cor(self.RD, query_image_path, self.patch_size, self.iter_patch_num, self.random_crop)
        return extreme_cor_dic, corner_cor_dic, ori_shape

