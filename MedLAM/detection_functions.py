#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
from os.path import abspath, join, dirname
sys.path.append(os.path.abspath(__file__))  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, join(abspath(dirname(__file__)), 'src'))
import  numpy as np
from data_process.data_process_func import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import morphology
import random
import copy
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class Relative_distance(object):
    def __init__(self,network,out_mode='fc_position', feature_refine_mode=False, distance_mode='linear', \
                center_patch_size=[8,8,8], distance_ratio=100):
        self.network = network
        self.center_patch_size = np.array(center_patch_size)
        self.distance_ratio = distance_ratio
        self.distance_mode = distance_mode
        self.out_mode = out_mode
        self.feature_refine_mode = feature_refine_mode
        self.support_label_position_dic = {}
        if self.feature_refine_mode:
            self.support_center_feature = {}
            self.support_center_conv = {}
            
    def cal_support_position(self, support_patch, support_label_patch=False, fg_class=1, idx=0):
        '''
        support_patch: [b*c*d*w*h]
        support_label_patch: [b*1*d*w*h]  
        '''
        if idx==0:
            self.support_center_feature[fg_class] = {}
        center_label_patch = crop_patch_around_center(support_label_patch, np.int8(self.center_patch_size//2)) #6,1,r,r,r 
        self.support_patch = support_patch
        self.full_shape = list(support_patch.shape[:1]+ support_patch.shape[2:])# 6, D, W, H
        self.support_all = self.network(x=torch.from_numpy(support_patch).float().half(), out_fc=True, out_feature=True, out_classification=False)
        if idx ==0:
            self.support_label_position_dic[fg_class] = self.support_all[self.out_mode].cpu().numpy()
        else:
            self.support_label_position_dic[fg_class] += self.support_all[self.out_mode].cpu().numpy()
        if self.feature_refine_mode == 'point':
            for key in self.support_all:
                if 'feature' in key:
                    support_feature = self.support_all[key] #[6,c,d,w,h]
                    shape = support_feature.shape
                    norm_feature = F.normalize(support_feature[:,:,shape[2]//2-1:shape[2]//2,shape[3]//2-1:shape[3]//2, \
                                                            shape[4]//2-1:shape[4]//2], dim=1) #[6,c,1,1,1]
                    if idx==0:
                        self.support_center_feature[fg_class][key] = norm_feature
                    else:
                        self.support_center_feature[fg_class][key] += norm_feature
        elif self.feature_refine_mode == 'corner':
            for key in self.support_all:
                if 'feature' in key:
                    support_feature = self.support_all[key] #[6,c,d,w,h]
                    zoom_factor = support_feature.shape[-1]/support_label_patch.shape[-1]
                    cur_support_center_label_patch = F.interpolate(center_label_patch.float(), scale_factor=zoom_factor, mode='trilinear')
                    cur_support_center_feature = crop_patch_around_center(support_feature, np.int8(self.center_patch_size//2*zoom_factor))
                    norm_feature = F.normalize(cur_support_center_feature, dim=1)*(cur_support_center_label_patch) #[6,c,r,r,r]
                    if idx==0:
                        self.support_center_feature[fg_class][key] = norm_feature
                    else:
                        self.support_center_feature[fg_class][key] += norm_feature
        
    def feature2conv(self, fg_class):
        self.support_center_conv[fg_class]={}
        for key in self.support_center_feature[fg_class].keys():
            groups = self.support_center_feature[fg_class][key].shape[0] # 6
            in_c = self.support_center_feature[fg_class][key].shape[1]*groups # 6c
            kernel_size = self.support_center_feature[fg_class][key].shape[2]
            self.support_center_conv[fg_class][key] = torch.nn.Conv3d(in_c, groups, kernel_size, 
                        padding=kernel_size//2, groups=groups).cuda().half() 
            self.support_center_conv[fg_class][key].weight = torch.nn.Parameter(self.support_center_feature[fg_class][key]) #[6,c,r,r,r]
            self.support_center_conv[fg_class][key] = self.support_center_conv[fg_class][key].half()
            

            
    def cal_RD(self, query_patch, fg_class, mean=False, out_fc=True, out_feature=False):
        '''
        query_patch:[b*1*d*w*h]
        '''
        result = {}
        query_all = self.network(x=torch.from_numpy(query_patch).float().half(), out_fc=out_fc, out_feature=out_feature, out_classification=False)
        quer_position = query_all[self.out_mode].cpu().numpy()# [b, 3]
        if self.out_mode == 'position':
            if self.center_patch_size =='all':
                quer_position = np.mean(quer_position, axis=(2,3,4)) #[6,3]
            else:
                quer_position = np.mean(crop_patch_around_center(quer_position, r=self.center_patch_size), axis=(2,3,4))
        if mean:
            quer_position = np.mean(quer_position, axis=0)
        if self.distance_mode=='linear':
            relative_position = self.distance_ratio*(self.support_label_position_dic[fg_class]-quer_position)
        elif self.distance_mode=='tanh':
            relative_position = self.distance_ratio*np.tanh(self.support_label_position_dic[fg_class]-quer_position)
        else:
            raise ValueError('Please select a correct distance mode!!！')
        result['relative_position']=relative_position
        return result
    
    def cal_RD_with_feature(self, query_patch, fg_class, spacing, out_fc=False, out_feature=True):
        '''
        query_patch:[b*1*d*w*h]
        '''
        result = {}
        query_all = self.network(x=torch.from_numpy(query_patch).float().half(),out_fc=out_fc,out_feature=out_feature, out_classification=False)
        cos_sim_map = np.ones(self.full_shape).astype(np.float16) # 6, d, w, h
        
        for key in query_all.keys():
            if 'feature' in key:
                norm_querry_feature = F.normalize(query_all[key], dim=1) # 6, c, d, w, h
                if self.feature_refine_mode == 'point':
                    cur_sim_map = torch.sum(norm_querry_feature*self.support_center_feature[key], dim=1, keepdim=True) # 6, 1, d, w, h
                elif self.feature_refine_mode == 'corner':
                    norm_querry_feature = norm_querry_feature.view(1, norm_querry_feature.shape[0]*norm_querry_feature.shape[1], \
                            norm_querry_feature.shape[2], norm_querry_feature.shape[3], norm_querry_feature.shape[2])
                    cur_sim_map = self.support_center_conv[fg_class][key](norm_querry_feature) # 6, 1, d, w, h
                    cur_sim_map /= self.support_center_conv[fg_class][key].weight.shape[-1]**3
                zoom_factor = cos_sim_map.shape[-1]/cur_sim_map.shape[-1]
                cur_sim_map = F.interpolate(cur_sim_map, scale_factor=zoom_factor, mode='trilinear').cpu().numpy().squeeze()
                cos_sim_map = cos_sim_map+cur_sim_map
                
        max_cor = []
        for i in range(cos_sim_map.shape[0]):
            cur_cor=np.array(np.where(cos_sim_map[i]==np.max(cos_sim_map[i])))
            max_cor.append(cur_cor[:, 0]) # 6, 3
        max_sim = np.max(cos_sim_map.reshape(cos_sim_map.shape[0], -1), axis=1)
        max_cor = np.array(max_cor)
        relative_cor = max_cor-np.array(self.full_shape[1::])[np.newaxis]//2+1 # 6,3
        result['cos_sim_map']=cos_sim_map
        result['relative_position']=relative_cor*spacing
        result['relative_cor'] = relative_cor
        result['max_sim']=max_sim
        return result
    


def random_all(random_seed):
    random.seed(random_seed)  
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def move(image, inher_position, test_patch_size):
    cur_position = np.asarray(image.shape)*np.asarray(inher_position)
    cur_position = cur_position.astype(np.int16)
    cur_image_patch = image[cur_position[0]:cur_position[0]+test_patch_size[0], cur_position[1]:cur_position[1]+test_patch_size[1],
                      cur_position[2]:cur_position[2]+test_patch_size[2]]
    return cur_image_patch

def get_center_cor(img):
    '''
    get 2d binary img center corardiate
    :param img: 2d binary img
    :return:
    '''
    mask = np.nonzero(img)
    half = len(mask[0])//2
    center_x = np.asarray(mask[0][half:half+1])
    center_y = np.asarray(mask[1][half:half+1])
    return center_x, center_y


def get_random_cor(img):
    '''
    randomly select one point from 2d binary img and return its coordinate
    :param img: 2d binary img
    :return:
    '''
    mask = np.nonzero(img)
    random_cor = random.randint(0, len(mask[0])-1)
    cor_x = np.asarray(mask[0][random_cor:random_cor+1])
    cor_y = np.asarray(mask[1][random_cor:random_cor+1])
    # cor_x = np.round(np.mean(np.asarray(mask[0])))
    # cor_y = np.round(np.mean(np.asarray(mask[1])))
    return cor_x, cor_y

def transfer_extremepoint_to_cornerpoint(extremepoint):
    cornerpoint = [[],[]]
    for i in range(3):  
        cornerpoint[0].append(extremepoint[i, i])
        cornerpoint[1].append(extremepoint[3+i, i]) 
    return np.asarray(cornerpoint)

def transfer_extremepoint_to_real_extremepoint(extremepoint):
    for i in range(3):
        if extremepoint[i, i]>extremepoint[3+i, i]:
            temp = copy.copy(extremepoint[i])
            extremepoint[i] = extremepoint[3+i]
            extremepoint[3+i] = temp   
    return extremepoint

def transfer_multi_extremepoint_to_cornerpoint(extremepoint, point_per_slice, slice_range):
    '''
    extremepoint: 6*3
    '''
    num = extremepoint.shape[0]
    cor_offset = np.arange(slice_range).repeat(point_per_slice)
    cornerpoint = [[],[]]
    for i in range(3):
        min_p = np.median(extremepoint[i*num//6: (i+1)*num//6, i]-cor_offset)
        max_p = np.median(extremepoint[(i+3)*num//6: (i+4)*num//6, i]+cor_offset)
        cornerpoint[0].append(min_p)
        cornerpoint[1].append(max_p)
    return np.asarray(cornerpoint)


def crop_patch_around_center(img, r):
    '''
    img: array, c*w*h*d
    r: list
    crop a patch around the center point with shape r
    '''
    if len(img.shape)==4:
        shape = img.shape[1::]
        patch = img[:, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==3:
        shape = img.shape
        patch = img[shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    elif len(img.shape)==5:
        shape = img.shape[2::]
        patch = img[:, :, shape[0]//2-r[0]:shape[0]//2+r[0], shape[1]//2-r[1]:shape[1]//2+r[1], shape[2]//2-r[2]:shape[2]//2+r[2]]
    return patch

def extract_fg_cor(label, extreme_point_num=2, erosion_num=False, pad=[0, 0, 0]):
    if erosion_num:
        label= morphology.binary_erosion(label, structure=np.ones([1,1,1]), iterations=erosion_num)
    extre_cor = get_bound_coordinate(label, pad=pad) #[minpoint, maxpoint]
    if extreme_point_num==2:
        return np.asarray(extre_cor)
    elif extreme_point_num==6:
        real_extre_point = np.zeros([6,3])
        for i in range(len(extre_cor)):
            for ii in range(len(extre_cor[i])):
                slice_label = label.transpose(ii,ii-2,ii-1)[extre_cor[i][ii]]
                #(center_x, center_y) = ndimage.center_of_mass(slice_label)
                center_x, center_y = get_center_cor(img=slice_label)
                cor=np.zeros(3)
                cor[ii]=extre_cor[i][ii]
                cor[ii-2]=center_x
                cor[ii-1]=center_y
                real_extre_point[i*3+ii] = cor
        return np.int16(real_extre_point)

def select_extreme_support_cor(label, point_per_slice=2, erosion_num=False, slice_range=2):
    if erosion_num:
        label= morphology.binary_erosion(label, structure=np.ones([1,1,1]), iterations=erosion_num)
    extre_cor = get_bound_coordinate(label) #[minpoint, maxpoint]
    support_cor = []
    for i in range(2): # 2
        for ii in range(3): # 3
            for iii in range(slice_range):
                slice_label = label.transpose(ii,ii-2,ii-1)[extre_cor[i][ii]+iii*(-1)**i]
                #(center_x, center_y) = ndimage.center_of_mass(slice_label)
                for _ in range(point_per_slice):
                    random_x, random_y = get_random_cor(img=slice_label)
                    cor=np.zeros(3)
                    cor[ii]=extre_cor[i][ii]+iii*(-1)**i
                    cor[ii-2]=random_x
                    cor[ii-1]=random_y
                    support_cor.append(cor)
    return np.array(support_cor, dtype=np.int16)

def cal_average_except_minmax(predicted_point_position, extract_m = False):
    

    predicted_point_position = np.asarray(predicted_point_position)
    position_each_axis = [predicted_point_position[:,i].tolist() for i in range(predicted_point_position.shape[1])]
    n = len(position_each_axis[0])//10
    for i in range(len(position_each_axis)):
        position_each_axis[i].sort()
        if not extract_m == False:
            position_each_axis[i] = position_each_axis[i][:-n]
            position_each_axis[i] = position_each_axis[i][n:]
        position_each_axis[i] = np.mean(position_each_axis[i])
    return np.asarray(position_each_axis)


def expand_cor_if_nessary(corner_cor, patch_size):
    w,h,d = corner_cor[1]-corner_cor[0]
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        corner_cor[0]-=np.asarray([wl_pad,hl_pad, dl_pad])
        corner_cor[1]+=np.asarray([wr_pad,hr_pad, dr_pad])  
    return corner_cor


def pad(img, pad_size):
    if len(img.shape)==3:
        img = np.pad(img, [(pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2),
                    (pad_size[2] // 2, pad_size[2] // 2)], mode='constant', constant_values=0)
    elif len(img.shape)==4:
        img = np.pad(img, [(0,0), (pad_size[0] // 2, pad_size[0] // 2), (pad_size[1] // 2, pad_size[1] // 2),
                    (pad_size[2] // 2, pad_size[2] // 2)], mode='constant', constant_values=0)
    return img

def crop(img, pad_size):
    img = img[ pad_size[0] // 2: -pad_size[0] // 2, pad_size[1] // 2: -pad_size[1] // 2,
                pad_size[2] // 2: -pad_size[2] // 2]
    return img

def iou(box1, box2):
    '3Diou,box=[h_min,w_min,d_min,h_max,w_max,d_max]'
    box1 = np.asarray(box1).reshape([-1,1])
    box2 = np.asarray(box2).reshape([-1,1])
    in_h = min(box1[3], box2[3]) - max(box1[0], box2[0])
    in_w = min(box1[4], box2[4]) - max(box1[1], box2[1])
    in_d =min(box1[5], box2[5]) - max(box1[2], box2[2])
    inter = 0 if in_h<0 or in_w<0 or in_d<0 else in_h*in_w*in_d
    union = (box1[3] - box1[0]) * (box1[4] - box1[1])*(box1[5] - box1[2]) + \
            (box2[3] - box2[0]) * (box2[4] - box2[1])*(box2[5] - box2[2]) - inter
    iou = inter / union
    return iou

class RandomPositionCrop(object):
    """
    random crop a patch from an image
    """

    def __init__(self, output_size, padding=True):
        self.output_size = output_size
        self.padding = padding
        self.max_outsize_each_axis = self.output_size 

    def random_cor(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(np.random.randint(shape[i]//2-10, shape[i]//2+10))
            #position.append(np.random.randint(self.max_outsize_each_axis[i]//2, shape[i] - self.max_outsize_each_axis[i]//2))
        return np.asarray(position)

    def __call__(self, sample):
        image,spacing= sample['image'],sample['spacing']
        # pad the sample if necessary
        if self.padding:
            image = np.pad(image, [(0,0), (self.max_outsize_each_axis[0]//2, self.max_outsize_each_axis[0]//2), (self.max_outsize_each_axis[1]//2,
                self.max_outsize_each_axis[1]//2), (self.max_outsize_each_axis[2]//2, self.max_outsize_each_axis[2]//2)], mode='constant', constant_values=0)
        if image.shape[1] <= self.max_outsize_each_axis[0] or image.shape[2] <= self.max_outsize_each_axis[1] or image.shape[3] <= \
                self.max_outsize_each_axis[2]:
            pw = max((self.max_outsize_each_axis[0] - image.shape[1]) // 2 + 3, 0)
            ph = max((self.max_outsize_each_axis[1] - image.shape[2]) // 2 + 3, 0)
            pd = max((self.max_outsize_each_axis[2] - image.shape[3]) // 2 + 3, 0)
            image = np.pad(image, [(0,0),(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        shape = image.shape[1:]
        background_chosen = True
        while background_chosen:
            random_cor = self.random_cor(shape)
            if image[0, random_cor[0] , random_cor[1] ,random_cor[2]] >= 0.001:
                background_chosen = False
        sample['random_position']=random_cor*np.asarray(spacing)
        image_patch = image[:, random_cor[0]-self.output_size[0]//2:random_cor[0] + self.output_size[0]//2,
                            random_cor[1]-self.output_size[1]//2:random_cor[1] + self.output_size[1]//2,
                            random_cor[2]-self.output_size[2]//2:random_cor[2] + self.output_size[2]//2]
        sample['random_crop_image']=image_patch
        return sample