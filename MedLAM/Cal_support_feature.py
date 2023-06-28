#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data_process.data_process_func import *
from MedLAM.detection_functions import *
import numpy as np
from tqdm import trange


def process_support(RD, support_volume_ls, support_label_ls, patch_size, fg_class=1):
    for idx in trange(len(support_volume_ls)):
        support_sample = load_and_pre_ct(support_volume_ls[idx])
        support_label = pad(load_and_pre_ct(support_label_ls[idx], mode='label')['image'], patch_size)
        support_label = 1*(support_label==fg_class)
        support_sample['image'] = pad(support_sample['image'], patch_size)
        support_extreme_cor = extract_fg_cor(support_label, 6, erosion_num=False)
        support_image_batch, support_label_batch = [], []
        for i in range(support_extreme_cor.shape[0]): 
            support_cor = support_extreme_cor[i]
            support_image_batch.append(support_sample['image'][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                        support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                        support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
            support_label_batch.append(support_label[support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                        support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                        support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
        support_image_batch = np.asarray(support_image_batch)
        support_label_batch = torch.tensor(np.array(support_label_batch)).cuda()
        RD.cal_support_position(support_image_batch, support_label_patch=support_label_batch, idx=idx, fg_class=fg_class)
    RD.support_label_position_dic[fg_class] /= len(support_volume_ls)
    RD.feature2conv(fg_class)
    return RD
