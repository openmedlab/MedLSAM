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
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import cv2
from skimage.measure import label, regionprops

def preprocess_ct(gt_path, nii_path, label_id, image_size, sam_model=None, device=0, gt_slice_threshold=0, z_min=None, z_max=None, padding=0):
    # do not select the z index
    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    gt_data = np.uint8(gt_data==label_id)
    if np.sum(gt_data)>10:
        imgs = {}
        gts =  {}
        img_embeddings = []
        assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'
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
        for i in range(z_min, z_max):
            gt_slice_i = gt_data[i,:,:]
            if np.sum(gt_slice_i)>gt_slice_threshold:
                img_slice_i = image_data_pre[i,:,:]
                gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
                # resize img_slice_i to image_sizeximage_size
                img_slice_i = transform.resize(img_slice_i, (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
                # convert to three channels
                
                img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                imgs[i] = img_slice_i
                # assert np.sum(gt_slice_i)>10, 'ground truth should have more than 10 pixels'
                gts[i] = gt_slice_i
                if sam_model is not None:
                    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                    resize_img = sam_transform.apply_image(img_slice_i)
                    # resized_shapes.append(resize_img.shape[:2])
                    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                    # model input: (1, 3, 1024, 1024)
                    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
                    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                    # input_imgs.append(input_image.cpu().numpy()[0])
                    with torch.no_grad():
                        embedding = sam_model.image_encoder(input_image)
                        img_embeddings.append(embedding.cpu().numpy()[0])

    if sam_model is not None:
        return imgs, gts, img_embeddings
    else:
        return imgs, gts