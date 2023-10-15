#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import nibabel as nib
import numpy as np
from nibabel.orientations import (apply_orientation, axcodes2ornt,
                                  inv_ornt_aff, ornt_transform)

from data_process.data_process_func import *
from MedLAM.detection_functions import *


def predic_corner_cor(RD, img_path, patch_size, iter_patch_num, random_crop, MSS):
    query_sample = load_and_pre_ct(img_path)
    spacing = 3
    query_sample['image'] = pad(query_sample['image'], patch_size)
    sss = query_sample['image'].shape
    query_sample['image'] = query_sample['image'][np.newaxis]
    start_position,query_batch = [],[]
    extreme_cor_dic = {}
    corner_cor_dic = {}
    for _ in range(iter_patch_num): # rancom crop several times, and average the result
        sample = random_crop(query_sample)
        random_position = np.int16(sample['random_position']).squeeze()
        start_position.append(random_position)
        query_batch.append(sample['random_crop_image'])
    query_batch = np.asarray(query_batch)
    query_position = RD.cal_query(query_batch=query_batch, mean=True, out_fc=True, out_feature=False)

    for fg_class in RD.support_label_position_dic.keys():
        relative_position = RD.cal_RD(query_position=query_position, fg_class=fg_class)['relative_position']
        # relative_position = RD.cal_RD_old(query_batch=query_batch, fg_class=fg_class, mean=True, out_fc=True, out_feature=False)['relative_position']
        cur_position = relative_position + np.mean(np.asarray(start_position), axis=0) # [b, 3]
        middle_query_batch = []
        extreme_cor = np.round(cur_position/spacing).astype(np.int16) # pixel coordinate
        if MSS:
            extreme_cor[:,0] = np.minimum(np.maximum(extreme_cor[:,0], patch_size[0]//2), sss[0]-patch_size[0]//2-1)
            extreme_cor[:,1] = np.minimum(np.maximum(extreme_cor[:,1], patch_size[1]//2), sss[1]-patch_size[1]//2-1)
            extreme_cor[:,2] = np.minimum(np.maximum(extreme_cor[:,2], patch_size[2]//2), sss[2]-patch_size[2]//2-1)
            for iii in range(cur_position.shape[0]):
                middle_query_batch.append(query_sample['image'][:, 
                            extreme_cor[iii,0] - patch_size[0]//2:extreme_cor[iii,0] + patch_size[0]//2,
                            extreme_cor[iii,1] - patch_size[1]//2:extreme_cor[iii,1] + patch_size[1]//2,
                            extreme_cor[iii,2] - patch_size[2]//2:extreme_cor[iii,2] + patch_size[2]//2])
            
            middle_query_batch = np.asarray(middle_query_batch) # [b, 1, D, W, H]
            result = RD.cal_RD_with_feature(query_patch=middle_query_batch, fg_class=fg_class, spacing=spacing, out_fc=False, out_feature=True)
            relative_cor = result['relative_cor']
            extreme_cor += relative_cor
        extreme_cor -= patch_size//2
        assert extreme_cor.shape[0]%6 == 0
        for i in range(extreme_cor.shape[0]//6):
            cur_extreme_cor = extreme_cor[i*6:(i+1)*6]
            corner_cor = transfer_extremepoint_to_cornerpoint(cur_extreme_cor)
            cur_extreme_cor = np.around(cur_extreme_cor*(1/query_sample['zoom_factor']))
            corner_cor = np.around(corner_cor*(1/query_sample['zoom_factor']))

            cur_extreme_cor = cur_extreme_cor[:, ::-1]
            corner_cor = corner_cor[:, ::-1]
            targ_ornt = axcodes2ornt('LAS')
            transform = ornt_transform(targ_ornt, query_sample['ori_ornt'])
            inv_ornt = inv_ornt_aff(transform, query_sample['ori_shape'])
            cur_extreme_cor_hom = np.concatenate([cur_extreme_cor, np.ones((cur_extreme_cor.shape[0], 1))], axis=-1)
            cur_extreme_cor_orig_hom = np.matmul(inv_ornt, cur_extreme_cor_hom.T).T
            cur_extreme_cor = cur_extreme_cor_orig_hom[:, :3]
            corner_cor_hom = np.concatenate([corner_cor, np.ones((corner_cor.shape[0], 1))], axis=-1)
            corner_cor_orig_hom = np.matmul(inv_ornt, corner_cor_hom.T).T
            corner_cor = corner_cor_orig_hom[:, :3]
            cur_extreme_cor = cur_extreme_cor[:, ::-1]
            corner_cor = corner_cor[:, ::-1]
            cur_extreme_cor = transfer_extremepoint_to_real_extremepoint(cur_extreme_cor)
            corner_cor = transfer_extremepoint_to_cornerpoint(cur_extreme_cor)
            if i ==0:
                extreme_cor_dic[fg_class]=[cur_extreme_cor.tolist()]
                corner_cor_dic[fg_class]=[corner_cor.tolist()]
            else:
                extreme_cor_dic[fg_class].append(cur_extreme_cor.tolist())
                corner_cor_dic[fg_class].append(corner_cor.tolist())
        

    return extreme_cor_dic, corner_cor_dic, query_sample['ori_shape']
