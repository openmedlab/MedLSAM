#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

sys.path.append(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import trange

from data_process.data_process_func import *
from MedLAM.detection_functions import *


def process_support(RD, support_volume_ls, support_label_ls, patch_size, fg_class_ls, bbox_mode='SPL', slice_interval=4):
    cur_support_image_ls, cur_support_label_ls = [], []
    z_length_ls = [[] for _ in range(len(fg_class_ls))]
    print('\n #### start to preprocess support volume #### \n ')
    for idx in trange(len(support_volume_ls)):
        support_image = pad(load_and_pre_ct(support_volume_ls[idx])['image'], patch_size)
        support_label = pad(load_and_pre_ct(support_label_ls[idx], mode='label')['image'], patch_size)
        # support_label = 1*(support_label==fg_class)
        for iidx in range(len(fg_class_ls)):
            bound_coor = get_bound_coordinate(1*(support_label==fg_class_ls[iidx]))
            z_length_ls[iidx].append(bound_coor[1,0]- bound_coor[0,0])
        cur_support_image_ls.append(support_image)
        cur_support_label_ls.append(support_label)
    print('\n #### start to extract support feature for each class #### \n')
    for iidx in trange(len(fg_class_ls)):
        for idx in range(len(support_volume_ls)):
            if bbox_mode == 'SPL': # Sub-Patch Localization
                if min(z_length_ls[iidx]) > slice_interval:
                    split_num = min(min(z_length_ls[iidx])//slice_interval, 6)
                    support_extreme_cor = extract_fg_cor(1*(cur_support_label_ls[idx]==fg_class_ls[iidx]), 'split', erosion_num=False, split_num=split_num)
                else:
                    support_extreme_cor = extract_fg_cor(1*(cur_support_label_ls[idx]==fg_class_ls[iidx]), 'extreme', erosion_num=False)
            elif bbox_mode == 'WPL': # Whole-Patch Localization
                support_extreme_cor = extract_fg_cor(1*(cur_support_label_ls[idx]==fg_class_ls[iidx]), 'extreme', erosion_num=False)
            support_image_batch, support_label_batch = [], []
            for i in range(support_extreme_cor.shape[0]): 
                support_cor = support_extreme_cor[i]
                
                support_image_batch.append(cur_support_image_ls[idx][support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                            support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                            support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
                support_label_batch.append(1*(cur_support_label_ls[idx]==fg_class_ls[iidx])[support_cor[0] - patch_size[0] // 2:support_cor[0] + patch_size[0] // 2,
                            support_cor[1] - patch_size[1] // 2:support_cor[1] + patch_size[1] // 2,
                            support_cor[2] - patch_size[2] // 2:support_cor[2] + patch_size[2] // 2][np.newaxis])
            support_image_batch = np.array(support_image_batch)
            support_label_batch = np.array(support_label_batch)
            RD.cal_support_position(support_image_batch, support_label_patch=support_label_batch, idx=idx, fg_class=fg_class_ls[iidx])
        RD.support_label_position_dic[fg_class_ls[iidx]] /= len(support_volume_ls)
        RD.feature2conv(fg_class_ls[iidx])
    return RD
