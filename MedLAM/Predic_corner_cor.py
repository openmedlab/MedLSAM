#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
sys.path.append(os.path.abspath(__file__))  #返回当前.py文件的绝对路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   #当前文件的绝对路径目录，不包括当前 *.py 部分，即只到该文件目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.data_process_func import *
from MedLAM.detection_functions import *
import numpy as np
import nibabel as nib
from nibabel.orientations import inv_ornt_aff, apply_orientation, ornt_transform, axcodes2ornt

def predic_corner_cor(RD, img_path, patch_size, iter_patch_num, random_crop):
    query_sample = load_and_pre_ct(img_path)
    spacing = 3
    query_sample['image'] = pad(query_sample['image'], patch_size)
    sss = query_sample['image'].shape
    query_sample['image'] = query_sample['image'][np.newaxis]
    start_position,query_batch = [],[]
    extreme_cor_dic = {}
    corner_cor_dic = {}
    for ii in range(iter_patch_num): # 多次随机裁减预测距离，最终取平均
        sample = random_crop(query_sample)
        random_position = np.int16(sample['random_position']).squeeze()
        start_position.append(random_position)
        query_batch.append(sample['random_crop_image'])
    query_batch = np.asarray(query_batch)
    for fg_class in RD.support_label_position_dic.keys():

        relative_position = RD.cal_RD(query_patch=query_batch, fg_class=fg_class, mean=True, out_fc=True, out_feature=False)['relative_position']
        cur_position = relative_position + np.mean(np.asarray(start_position), axis=0) # [6, 3]
        
        middle_query_batch = []
        cur_cor = np.round(cur_position/spacing).astype(np.int16) #像素坐标
        cur_cor[:,0] = np.minimum(np.maximum(cur_cor[:,0], patch_size[0]//2), sss[0]-patch_size[0]//2-1)
        cur_cor[:,1] = np.minimum(np.maximum(cur_cor[:,1], patch_size[1]//2), sss[1]-patch_size[1]//2-1)
        cur_cor[:,2] = np.minimum(np.maximum(cur_cor[:,2], patch_size[2]//2), sss[2]-patch_size[2]//2-1)
        
        for iii in range(cur_position.shape[0]):
            middle_query_batch.append(query_sample['image'][:, 
                        cur_cor[iii,0] - patch_size[0] // 2:cur_cor[iii,0] + patch_size[0] // 2,
                        cur_cor[iii,1] - patch_size[1] // 2:cur_cor[iii,1] + patch_size[1] // 2,
                        cur_cor[iii,2] - patch_size[2] // 2:cur_cor[iii,2] + patch_size[2] // 2])
        
        middle_query_batch = np.asarray(middle_query_batch) # [6, 1, D, W, H]
        result = RD.cal_RD_with_feature(query_patch=middle_query_batch, fg_class=fg_class, spacing=spacing, out_fc=False, out_feature=True)
        relative_cor = result['relative_cor']
        extreme_cor = (cur_cor+ relative_cor) - patch_size//2
        
        
        corner_cor = transfer_extremepoint_to_cornerpoint(extreme_cor)
        extreme_cor = np.around(extreme_cor*(1/query_sample['zoom_factor']))
        corner_cor = np.around(corner_cor*(1/query_sample['zoom_factor']))

        extreme_cor = extreme_cor[:, ::-1]
        corner_cor = corner_cor[:, ::-1]
        targ_ornt = axcodes2ornt('LAS')
        transform = ornt_transform(targ_ornt, query_sample['ori_ornt'])
        inv_ornt = inv_ornt_aff(transform, query_sample['ori_shape'])
        extreme_cor_hom = np.concatenate([extreme_cor, np.ones((extreme_cor.shape[0], 1))], axis=-1)
        extreme_cor_orig_hom = np.matmul(inv_ornt, extreme_cor_hom.T).T
        extreme_cor = extreme_cor_orig_hom[:, :3]
        corner_cor_hom = np.concatenate([corner_cor, np.ones((corner_cor.shape[0], 1))], axis=-1)
        corner_cor_orig_hom = np.matmul(inv_ornt, corner_cor_hom.T).T
        corner_cor = corner_cor_orig_hom[:, :3]
        extreme_cor = extreme_cor[:, ::-1]
        corner_cor = corner_cor[:, ::-1]
        extreme_cor = transfer_extremepoint_to_real_extremepoint(extreme_cor)
        corner_cor = transfer_extremepoint_to_cornerpoint(extreme_cor)
        extreme_cor_dic[fg_class]=extreme_cor.tolist()
        corner_cor_dic[fg_class]=corner_cor.tolist()
        

    return extreme_cor_dic, corner_cor_dic, query_sample['ori_shape']
