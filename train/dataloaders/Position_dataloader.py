#!/usr/bin/env python
# -*- coding: utf-8 -*-
from operator import is_not
import os
from numpy.lib.function_base import append
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import random
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *
import torchio as tio
from multiprocessing import Pool, cpu_count
import tqdm
import copy

def create_sample(image_path, out_size):
    
    image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
    image = image.astype(np.float16)
    if len(image.shape)==3:
        image = image[np.newaxis,:]
    elif len(image.shape) !=4:
        raise(ValueError(image_path))
    spacing = np.array(spacing).astype(np.float32)
    spacing = spacing[:,np.newaxis,np.newaxis,np.newaxis]
    if out_size is not None:
        cur_shape = image.shape[-3:]
        if cur_shape[0] <= out_size[0] or cur_shape[1] <= out_size[1] or cur_shape[2] <= \
                out_size[2]:
            pw = max(out_size[0] - cur_shape[0], 0)
            ph = max(out_size[1] - cur_shape[1], 0)
            pd = max(out_size[2] - cur_shape[2], 0)
            image = np.pad(image, [(0,0), (0, pw), (0, ph), (0, pd)], mode='constant', constant_values=0)
    sample = {'image': torch.from_numpy(image), 'spacing':torch.from_numpy(spacing),'image_path':image_path}
    return sample
        

class PositionDataloader(Dataset):
    """ Dataset position """
    def __init__(self, iter_num=None, image_list=False, num=None, transform=None, 
                random_sample=True, load_memory=True, parral_load = False, 
                out_size=None, crop_pad=0, batch_size=1):
        self._iternum = iter_num
        self.out_size = np.array(out_size)+2*np.array(crop_pad)
        self.transform = transform
        self.sample_list = []
        self.image_dic = {}
        image_task_dic = {}
        self.batch_size = batch_size
        self.load_memory = load_memory
        self.random_sample = random_sample
        self.image_list = read_file_list(image_list)
        if load_memory:
            if parral_load:
                p = Pool(cpu_count())
                for i in tqdm.trange(len(self.image_list)):
                    image_name = self.image_list[i]
                    image_task_dic[image_name]= p.apply_async(create_sample, args=(image_name, self.out_size,))
                p.close()
                p.join()
                for image_name in image_task_dic.keys():
                    self.image_dic[image_name]=image_task_dic[image_name].get()
            else:
                for i in tqdm.trange(len(self.image_list)):
                    image_name = self.image_list[i]
                    self.image_dic[image_name]= create_sample(image_name, self.out_size)
        
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))


    def __len__(self):
        if self.random_sample:
            return self._iternum*self.batch_size
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        
        if self.load_memory:
            sample = self.image_dic[random.sample(self.image_list, 1)[0]].copy()
        else:
            if self.random_sample:
                image_name = random.sample(self.image_list, 1)[0]
            else:
                image_name = self.image_list[idx]
            sample = create_sample(image_name, self.out_size)
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomDoubleCrop(object):
    """
    Randomly crop several images in one sample;
    distance is a vector(could be positive or pasitive), representing the vector
    from image1 to image2.
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, foreground_only=True, small_move=False, fluct_range=[0,0,0], crop_pad=0):
        self.output_size = torch.tensor(output_size, dtype=torch.int16)+2*torch.tensor(crop_pad, dtype=torch.int16)
        self.img_pad = torch.tensor(self.output_size)//2
        self.foregroung_only = foreground_only
        self.fluct_range = fluct_range # distance, mm
        self.small_move = small_move
        cor_x,cor_z,cor_y = np.meshgrid(np.arange(self.output_size[1]), 
                                        np.arange(self.output_size[0]), 
                                        np.arange(self.output_size[2]))
        self.cor_grid = torch.from_numpy(np.concatenate((cor_z[np.newaxis], \
                                                        cor_x[np.newaxis], \
                                                        cor_y[np.newaxis]), axis=0))

    def random_position(self, shape, initial_position=[0,0,0], spacing=[1,1,1], small_move=False):
        position = []
        for i in range(len(shape)):
            if small_move:
                position.append(random.randint(max(0, initial_position[i]-np.int(self.fluct_range[i]/spacing[i])), \
                                               min(shape[i]-1, initial_position[i]+np.int(self.fluct_range[i]/spacing[i]))))
            else:
                position.append(random.randint(0, shape[i]-1))
        return torch.tensor(position)

    def __call__(self, sample):
        image= sample['image']
        nsample ={}
        nsample['image_path']=sample['image_path']
        nsample['spacing']=sample['spacing']
        spacing = sample['spacing'].squeeze().numpy()
        
        background_chosen = True
        shape_n = torch.tensor(image.shape[1::])
        choose_num = 0
        while background_chosen:
            choose_num +=1
            random_pos0 = self.random_position(shape_n)
            if image[0, random_pos0[0], random_pos0[1], random_pos0[2]]>=0.001 or choose_num>=20:
                background_chosen = False
        pad_size_ = [torch.maximum(-random_pos0+self.output_size//2, torch.tensor(0)).to(torch.int16), 
                     torch.maximum(random_pos0+self.output_size//2-shape_n, torch.tensor(0)).to(torch.int16)]
        pad_size = []
        for i in range(3):
            pad_size.append(pad_size_[0][2-i])
            pad_size.append(pad_size_[1][2-i])
        min_random_pos0 = torch.maximum(random_pos0-self.output_size//2, torch.tensor(0)).to(torch.int16)
        max_random_pos0 = torch.minimum(random_pos0+self.output_size//2, shape_n).to(torch.int16)
        
        nsample['random_crop_image_0']=F.pad(image[:, min_random_pos0[0]:max_random_pos0[0],
                                                    min_random_pos0[1]:max_random_pos0[1],
                                                    min_random_pos0[2]:max_random_pos0[2]], pad = pad_size)
        
        nsample['random_position_0'] = torch.tensor(random_pos0)
        nsample['random_fullsize_position_0'] = self.cor_grid + torch.tensor(random_pos0).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        
        background_chosen = True
        choose_num = 0
        while background_chosen:
            choose_num +=1
            random_pos1= self.random_position(shape_n, nsample['random_position_0'], spacing, self.small_move)
            if image[0, random_pos1[0], random_pos1[1], random_pos1[2]]>=0.001 or choose_num>=20:
                background_chosen = False
        pad_size_ = [torch.maximum(-random_pos1+self.output_size//2, torch.tensor(0)).to(torch.int16), 
                     torch.maximum(random_pos1+self.output_size//2-shape_n, torch.tensor(0)).to(torch.int16)]
        pad_size = []
        for i in range(3):
            pad_size.append(pad_size_[0][2-i])
            pad_size.append(pad_size_[1][2-i])
        min_random_pos1 = torch.maximum(random_pos1-self.output_size//2, torch.tensor(0)).to(torch.int16)
        max_random_pos1 = torch.minimum(random_pos1+self.output_size//2, shape_n).to(torch.int16)
        nsample['random_crop_image_1']=F.pad(image[:, min_random_pos1[0]:max_random_pos1[0],
                                                    min_random_pos1[1]:max_random_pos1[1],
                                                    min_random_pos1[2]:max_random_pos1[2]], pad=pad_size)
        nsample['random_position_1'] = torch.tensor(random_pos1)
        nsample['random_fullsize_position_1'] = self.cor_grid + torch.tensor(random_pos1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        key_ls = list(nsample.keys())
        for key in key_ls:
            if torch.is_tensor(nsample[key]):
                nsample[key]=nsample[key].to(torch.float32)
                nsample[key.replace('random', 'ori_random')]=copy.deepcopy(nsample[key])
        return nsample

class RandomDoubleNoise(object):
    def __init__(self, mean=0, std=0.1,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_noise_0 = tio.RandomNoise(mean=mean, std=std, include=include_0)
        self.add_noise_1 = tio.RandomNoise(mean=mean, std=std, include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.add_noise_0(sample)
        if torch.rand(1)<self.prob:
            sample= self.add_noise_1(sample)
        return sample

class RandomDoubleIntensity(object):
    def __init__(self,include_0=['random_crop_image_0'], 
                 include_1=['random_crop_image_1'], k=0.1, hu_ls=[0, 0.25, 0.5, 0.75, 1], prob=0):
        self.prob = prob
        self.include_0 = include_0
        self.include_1 = include_1
        self.k = k
        self.hu_ls = hu_ls
        
    def renorm(self, sample, include):
        for key in include:
            new_file=torch.zeros_like(sample[key], dtype=torch.float32)
            for i in range(len(self.hu_ls)-1):
                random_hu = np.random.uniform(self.hu_ls[i], self.hu_ls[i+1])
                new_file += 1/((len(self.hu_ls)-1)*(1+torch.exp((-sample[key]+random_hu)/0.1)))
            sample[key] = new_file
        return sample
    
    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample= self.renorm(sample, self.include_0)
        if torch.rand(1)<self.prob:
            sample= self.renorm(sample, self.include_1)
        return sample

class RandomDoubleFlip(object):
    def __init__(self, include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.flip_probability = prob
        self.include0 = include_0
        self.include1 = include_1
    def __call__(self, sample):
        axes = np.random.randint(0, 2)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include0)
        axes = np.random.randint(0, 2)
        flip = tio.RandomFlip(axes=axes, flip_probability=self.flip_probability, include = self.include1)
        sample= flip(sample)
        return sample

class RandomDoubleSpike(object):
    def __init__(self, num_spikes=3, intensity=1.2,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_spike_0 = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include_0)
        self.add_spike_1 = tio.RandomSpike(num_spikes=num_spikes, intensity=intensity,include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_spike_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_spike_1(sample)
        return sample

class RandomDoubleGhosting(object):
    def __init__(self, intensity=0.8,include_0=['random_crop_image_0'], include_1=['random_crop_image_1'], prob=0):
        self.prob = prob
        self.add_ghost_0 = tio.RandomGhosting(intensity=intensity, include=include_0)
        self.add_ghost_1 = tio.RandomGhosting(intensity=intensity, include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_ghost_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_ghost_1(sample)
        return sample

class RandomDoubleElasticDeformation(object):
    def __init__(self, num_control_points=[5,10,10], max_displacement=[7,7,7],
                include_0=['random_crop_image_0','random_fullsize_position_0'], 
                include_1=['random_crop_image_1','random_fullsize_position_1'], prob=0):
        self.prob = prob
        self.add_elas_0 = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            include=include_0)
        self.add_elas_1 = tio.RandomElasticDeformation(
            num_control_points=num_control_points,
            max_displacement = max_displacement,
            include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_elas_1(sample)
        return sample

class RandomDoubleAffine(object):
    def __init__(self, scales=[0.2,0.2,0.2], degrees=[10,10,10],
                include_0=['random_crop_image_0','random_fullsize_position_0'], 
                include_1=['random_crop_image_1','random_fullsize_position_1'], prob=0):
        self.prob = prob
        self.add_elas_0 = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include_0)
        self.add_elas_1 = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include_1)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas_0(sample)
        if torch.rand(1)<self.prob:
            sample=self.add_elas_1(sample)
        return sample
    
class RandomAffine(object):
    def __init__(self, scales=[0.2,0.2,0.2], degrees=[10,10,10],
                include=['random_crop_image_0','random_fullsize_position_0', 
                           'random_crop_image_1','random_fullsize_position_1'], prob=0):
        self.prob = prob
        self.add_elas = tio.RandomAffine(
            scales=scales,
            degrees=degrees,
            include=include)

    def __call__(self, sample):
        if torch.rand(1)<self.prob:
            sample=self.add_elas(sample)
        return sample

class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, crop_pad=0, output_size=0):
        self.crop_pad = crop_pad
        self.output_size = output_size
        
    def random_position(self, shape):
        position = []
        for i in range(len(shape)):
            position.append(random.randint(0, shape[i] - self.output_size[i]))
        return position
    
    def __call__(self, sample):
        cshape = sample['random_fullsize_position_0'].shape[1::]
        random_pos0 = self.random_position(cshape)
        random_pos1 = self.random_position(cshape)
        for key in sample.keys():
            if isinstance(sample[key], torch.Tensor):
                if sample[key].shape[1::]==cshape:
                    if 'ori' not in key:
                        sample[key] = sample[key][:, self.crop_pad:cshape[0]-self.crop_pad, \
                                self.crop_pad:cshape[1]-self.crop_pad, self.crop_pad:cshape[2]-self.crop_pad] #Prevent the black edge at the image boundary after deformation
                    elif key.endswith('0') or key.endswith('1'):
                        sample[key] = sample[key][:, random_pos0[0]:random_pos0[0]+self.output_size[0], \
                                                    random_pos0[1]:random_pos0[1]+self.output_size[1], \
                                                    random_pos0[2]:random_pos0[2]+self.output_size[2]]
        shape = sample['random_fullsize_position_0'].shape
        spacing = sample['spacing'] #[3,1,1,1]
        sample['random_fullsize_position_0'] *= spacing
        sample['random_fullsize_position_1'] *= spacing
        sample['random_position_0']=sample['random_fullsize_position_0'][:,shape[1]//2, shape[2]//2, shape[3]//2] #[3]
        sample['random_position_1']=sample['random_fullsize_position_1'][:,shape[1]//2, shape[2]//2, shape[3]//2]
        sample['rela_distance']=sample['random_position_0']-sample['random_position_1']
        sample['ori_random_fullsize_position_0'] *= spacing
        sample['ori_random_fullsize_position_1'] *= spacing
        sample['ori_random_position_0']=sample['ori_random_fullsize_position_0'][:,shape[1]//2, shape[2]//2, shape[3]//2]
        sample['ori_random_position_1']=sample['ori_random_fullsize_position_1'][:,shape[1]//2, shape[2]//2, shape[3]//2]
        sample['ori_rela_distance'] = sample['ori_random_position_0']-sample['ori_random_position_1']
        return sample

