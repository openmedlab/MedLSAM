#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 6 19:25:36 2023

convert CT nii image to npz files, including input image, image embeddings, and ground truth

@author: jma
"""
#%% import packages
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform
from skimage.measure import label, regionprops
from concurrent.futures import ThreadPoolExecutor
import concurrent
from data_process.data_process_func import *

def process_slice(i, z_min, z_max, gt_data, image_data_pre, image_size, gt_slice_threshold, sam_model, label_id_ls):
    if i < z_min or i >= z_max:
        return None
    
    gt_slice_i = gt_data[i,:,:]
    if np.sum(1*(gt_slice_i>0)) <= gt_slice_threshold:
        return None

    img_slice_i = image_data_pre[i,:,:]
    gt_slice_i = resize_Multi_label_to_given_shape(gt_slice_i, target_shape=image_size).numpy()
    gt_slice_i = labeltrans(label_id_ls, gt_slice_i)
    img_slice_i = transform.resize(img_slice_i, (image_size, image_size), order=2, preserve_range=True, mode='constant', anti_aliasing=True)
    img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))

    # 如果有 SAM 模型，则进行额外处理
    img_embedding = None
    if sam_model is not None:
        # 这里添加针对 SAM 模型的处理，例如：
        img_embedding = sam_model.process(img_slice_i)

    return i, img_slice_i, gt_slice_i, img_embedding

def preprocess_ct(gt_path, nii_path, label_id_ls, image_size, sam_model=None, device=0, gt_slice_threshold=0, z_min=None, z_max=None, padding=0):
    # do not select the z index
    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = sitk.GetArrayFromImage(gt_sitk).astype(np.int16)
    label_id_pair = [[label_id_ls[i], i+1] for i in range(len(label_id_ls))]
    reverse_label_id_pair = [[i+1, label_id_ls[i]] for i in range(len(label_id_ls))]
    gt_data = labeltrans(label_id_pair, gt_data)
    if np.sum(1*(gt_data>0))>10:
        imgs = {}
        gts =  {}
        img_embeddings = []
        img_sitk = sitk.ReadImage(nii_path)
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        lower_bound = -500
        upper_bound = 1000
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.int32(image_data_pre)
        z_index, _, _ = np.where(gt_data>0)
        if z_min is None:
            z_min = np.min(z_index)
        else:
            z_min = np.int32(max(0, z_min-padding))
        if z_max is None:
            z_max = np.max(z_index)
        else:
            z_max = np.int32(min(gt_data.shape[0], z_max+padding))

        with ThreadPoolExecutor(16) as executor:
            futures = [executor.submit(process_slice, i, z_min, z_max, gt_data, image_data_pre, image_size, gt_slice_threshold, sam_model, reverse_label_id_pair) for i in range(gt_data.shape[0])]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    i, img_slice_i, gt_slice_i, img_embedding = result
                    imgs[i] = img_slice_i
                    gts[i] = gt_slice_i
                    if img_embedding is not None:
                        img_embeddings.append(img_embedding)

    if sam_model is not None:
        return imgs, gts, img_embeddings
    else:
        return imgs, gts