#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import nibabel
import numpy as np
from scipy import ndimage
import torch
from nibabel.orientations import axcodes2ornt, aff2axcodes, ornt2axcodes, flip_axis, ornt_transform

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def save_array_as_nifty_volume(data, filename, transpose=True, pixel_spacing=[1, 1, 3]):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = nibabel.Nifti1Image(data, None)
    img.header.set_zooms(pixel_spacing)
    nibabel.save(img, filename)

def convert_label(in_volume, label_convert_source, label_convert_target):
    """
    convert the label value in a volume
    inputs:
        in_volume: input nd volume with label set label_convert_source
        label_convert_source: a list of integers denoting input labels, e.g., [0, 1, 2, 4]
        label_convert_target: a list of integers denoting output labels, e.g.,[0, 1, 2, 3]
    outputs:
        out_volume: the output nd volume with label set label_convert_target
    """
    mask_volume = np.zeros_like(in_volume)
    convert_volume = np.zeros_like(in_volume)
    for i in range(len(label_convert_source)):
        source_lab = label_convert_source[i]
        target_lab = label_convert_target[i]
        if (source_lab != target_lab):
            temp_source = np.asarray(in_volume == source_lab)
            temp_target = target_lab * temp_source
            mask_volume = mask_volume + temp_source
            convert_volume = convert_volume + temp_target
    out_volume = in_volume * 1
    out_volume[mask_volume > 0] = convert_volume[mask_volume > 0]
    return out_volume

def reorient(img, targ_axcode="RAS"):
    """
    Reorient an NifTiImage to a target orientation (RAS by default).

    return: the reoriented NifTiImage object.
    """

    ori_ornt = nibabel.io_orientation(img.affine)
    targ_ornt = axcodes2ornt(targ_axcode)
    if (ori_ornt == targ_ornt).all():
        return img, ori_ornt
    else:
        print("The image is not in the target orientation. Reorienting...")
        transform = ornt_transform(ori_ornt, targ_ornt)

        img_orient = img.as_reoriented(transform)

        return img_orient, ori_ornt

def get_bound_coordinate(file, pad=[0, 0, 0]):
    '''
    输出array非0区域的各维度上下界坐标+-pad
    :param file: groundtruth图,
    :param pad: 各维度扩充的大小
    :return: bound: [min,max]
    '''
    file_size = file.shape
    nonzeropoint = np.asarray(np.nonzero(file))  # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
    maxpoint = np.max(nonzeropoint, 1).tolist()
    minpoint = np.min(nonzeropoint, 1).tolist()
    for i in range(len(pad)):
        maxpoint[i] = min(maxpoint[i] + pad[i], file_size[i]-1)
        minpoint[i] = max(minpoint[i] - pad[i], 0)
    return np.array([minpoint, maxpoint])

def divide_and_get_bounds(file, direction, num_parts, pad=[0, 0, 0]):
    bounds = []
    minpoint, maxpoint = get_bound_coordinate(file, pad=pad)
    step = (maxpoint[direction] - minpoint[direction]) // num_parts
    
    for i in range(num_parts):
        start = minpoint[direction] + i * step
        end = start + step if i < num_parts - 1 else maxpoint[direction]
        cur_file = file.copy()
        if direction == 0:
            cur_file[:start, :, :] = 0
            cur_file[end:, :, :] = 0
        elif direction == 1:
            cur_file[:, :start, :] = 0
            cur_file[:, end:, :] = 0
        elif direction == 2:
            cur_file[:, :, :start] = 0
            cur_file[:, :, end:] = 0

        if cur_file.sum() >0: 
            current_minpoint, current_maxpoint = get_bound_coordinate(cur_file, pad=pad)

        current_minpoint[direction] = start
        current_maxpoint[direction] = end
        
        bounds.append([current_minpoint, current_maxpoint])
    
    return np.array(bounds)




def labeltrans(labelpair, file):
    '''
    :param labelpair: labelpair list [[], ..., []]
    :param file: np array
    :return:
    '''
    newfile = np.zeros_like(file)
    for label in labelpair:
        newfile[np.where(file == label[0])] = label[1]
    return newfile


def load_and_pre_ct(image_path, mode='image'):
    if image_path.endswith('.nii.gz') or image_path.endswith('nii'):
        data = nibabel.load(image_path)
        ori_shape = data.shape
        data, ori_ornt = reorient(data, targ_axcode="LAS")
        image = data.get_fdata()
        spacing = list(data.header.get_zooms())
        image = image.transpose(2, 1, 0)
        spacing.reverse()
    else:
        raise ValueError('Must be nifti file!!!')
    
    if spacing != [3,3,3]:
        zoom_factor = np.array(spacing) / np.array([3, 3, 3])
        if mode == 'image':
            image = ndimage.zoom(image, zoom=zoom_factor)
        elif mode =='label':
            image = np.int32(resize_Multi_label_to_given_shape(np.int32(image), zoom_factor=zoom_factor).numpy())
        else:
            raise ValueError('mode must be image or label')
        spacing = [3, 3, 3]
    if mode == 'image':
        if image.min()<-10:
            image=img_multi_thresh_normalized(image, thresh_lis=[-1000, -200, 200, 1500]
                                            , norm_lis=[0,0.2,0.8,1])
        else:
            print('Please make sure you have normalized the CT file correctly!!!')
    sample = {'image': image, 'spacing':spacing, 'image_path':image_path, 'ori_shape':ori_shape, 'ori_ornt': ori_ornt, 'zoom_factor': zoom_factor}

    return sample


def img_multi_thresh_normalized(file, thresh_lis=[0], norm_lis=[0], data_type=np.float32):
    """
    this Segmental Linear Normalization Function was proposed in "Automatic Segmentation of Organs-at-Risk from Head-and-Neck 
    CT using Separable Convolutional Neural Network with Hard-Region-Weighted Loss"
    """
    # Creates an all-zero array of the same size as the file
    new_file = np.zeros_like(file, dtype=data_type)+norm_lis[0]
    thresh_lis = np.array(thresh_lis)
    norm_lis = np.array(norm_lis)
    
    # The slope and intercept between each threshold were calculated
    slopes = (norm_lis[1:] - norm_lis[:-1]) / (thresh_lis[1:] - thresh_lis[:-1])
    intercepts = norm_lis[:-1]
    
    # For each threshold interval, a broadcast was used to calculate the result
    for i in range(len(thresh_lis) - 1):
        mask = (file >= thresh_lis[i]) & (file < thresh_lis[i + 1])
        new_file[mask] = slopes[i] * (file[mask] - thresh_lis[i]) + intercepts[i]
    
    # For all elements larger than the last threshold, a value is assigned directly
    new_file[file >= thresh_lis[-1]] = norm_lis[-1]
    
    return new_file


def resize_Multi_label_to_given_shape(volume, zoom_factor=None, target_shape=None):
    """
    resize an multi class label to a given shape
    :param volume: the input label, an tensor
    :param zoom_factor: the zoom fatcor of z,x,y
    :param class_number: the number of classes
    :param order:  the order of the interpolation
    :return:   shape = zoom_factor*original shape z,x,y
    """
    if torch.is_tensor(volume):
        volume = volume.long()
    else:
        volume = torch.from_numpy(volume.copy()).long()
    if volume.dim() == 3:
        oh_volume = torch.nn.functional.one_hot(volume, -1).float().permute(3,0,1,2).unsqueeze(0)
        if zoom_factor is not None:
            output = torch.nn.functional.interpolate(oh_volume, scale_factor=list(zoom_factor), mode='trilinear')
        elif target_shape is not None:
            output = torch.nn.functional.interpolate(oh_volume, size=target_shape, mode='trilinear')
        else:
            raise ValueError('zoom_factor or target_shape must be given')
    elif volume.dim() == 2:
        oh_volume = torch.nn.functional.one_hot(volume, -1).float().permute(2,0,1).unsqueeze(0)
        if zoom_factor is not None:
            output = torch.nn.functional.interpolate(oh_volume, scale_factor=list(zoom_factor), mode='bilinear')
        elif target_shape is not None:
            output = torch.nn.functional.interpolate(oh_volume, size=target_shape, mode='bilinear')
        else:
            raise ValueError('zoom_factor or target_shape must be given')
    
    output = torch.argmax(output, dim=1).data.squeeze()
    return output

def load_nifty_volume_as_array(filename, transpose=True, return_spacing=False, respacing=False, target_spacing=1,
                               mode='image', order=3):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
    """
    img = nibabel.load(filename)
    data = img.get_fdata()
    spacing = list(img.header.get_zooms())
    if transpose:
        data = data.transpose(2, 1, 0)
        spacing.reverse()
    if respacing:
        zoomfactor = list(np.array(spacing) / np.array(target_spacing))
        spacing = target_spacing
        if mode == 'image':
            data = ndimage.zoom(data, zoom=zoomfactor, order=order)
        elif mode == 'label':
            data = np.int8(data)
            data = np.int8(resize_Multi_label_to_given_shape(data, zoom_factor=zoomfactor))
        else:
            ValueError('Please choose the right data mode! ( \'label\', or \'image\')')
    if return_spacing:
        return data, spacing
    else:
        return data