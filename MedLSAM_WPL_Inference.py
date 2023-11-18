# %% load environment
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import numpy as np

join = os.path.join
import argparse
import json
import random
import time
import traceback

import cv2
import torch
from tqdm import trange

from data_process.data_process_func import *
from MedLAM.Anatomy_detection import AnatomyDetection
from MedLAM.detection_functions import *
from MedSAM.auto_pre_CT import *
from MedSAM.segment_anything.build_sam import sam_model_registry
from MedSAM.segment_anything.utils.transforms import ResizeLongestSide
from util.evaluation_index import hd95
from util.parse_config import parse_config


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))    


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum, volume_intersect, volume_sum

def finetune_model_predict(img_np, box_np, sam_trans, sam_model_tune, device='cuda:0'):
    H, W = img_np.shape[:2]
    resize_img = sam_trans.apply_image(img_np)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
    input_image = sam_model_tune.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(input_image.to(device)) # (1, 256, 64, 64)
        # convert box to 1024x1024 grid
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] # (B, 1, 4)
        
        sparse_embeddings, dense_embeddings = sam_model_tune.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        medsam_seg_prob, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 256, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        medsam_seg_prob = torch.sigmoid(medsam_seg_prob)
        # reshape to original size
        medsam_seg_prob = torch.nn.functional.interpolate(medsam_seg_prob, size=(H, W), mode='bilinear', align_corners=False)
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
    return medsam_seg, medsam_seg_prob

#% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedLSAM')
parser.add_argument('-c', '--config_file', type=str, help='path to the config file')
args = parser.parse_args()
config_file = parse_config(args.config_file)

#% load MedLSAM model. Apply Sub-Patch Localization (SPL)
ana_det = AnatomyDetection(args.config_file)
sam_model_tune = sam_model_registry[config_file['vit']['net_type']](checkpoint=config_file['weight']['vit_load_path']).to('cuda:0')
sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

nii_pathes = read_file_list(config_file['data']['query_image_ls'])
gt_pathes = read_file_list(config_file['data']['query_label_ls'])
os.makedirs(config_file['data']['seg_png_save_path'], exist_ok=True)
os.makedirs(config_file['data']['seg_save_path'], exist_ok=True)
os.makedirs(f'{current_directory}/result/json', exist_ok=True)
os.makedirs(f'{current_directory}/result/dsc', exist_ok=True)
os.makedirs(f'{current_directory}/result/hd95', exist_ok=True)

sam_dice_scores = {key:[] for key in config_file['data']['fg_class']}
sam_hd95_scores = {key:[] for key in config_file['data']['fg_class']}

print('\n #### start inference ####')
for id in trange(len(nii_pathes)):
    nii_path = nii_pathes[id]
    gt_path = gt_pathes[id]
    time0 = time.time()
    extreme_cor_dic, corner_cor_dic, ori_shape = ana_det.get_extreme_corner(nii_path)
    time1 = time.time()
    print('get extreme corner time: ', time1 - time0)
    z_min_ls = [corner_cor_dic[key][0][0][0] for key in corner_cor_dic.keys()]
    z_max_ls = [corner_cor_dic[key][-1][1][0] for key in corner_cor_dic.keys()]
    z_min = min(z_min_ls) # minimum z of all organs
    z_max = max(z_max_ls) # maximum z of all organs
    time2 = time.time()
    imgs, gts = preprocess_ct(gt_path, nii_path, label_id_ls=list(corner_cor_dic.keys()), image_size=1024, \
                    gt_slice_threshold=config_file['data']['gt_slice_threshold'], z_min=z_min, z_max=z_max, padding=0)
    time3 = time.time()
    gt_sitk = sitk.ReadImage(gt_path)
    gt_data = sitk.GetArrayFromImage(gt_sitk).astype(np.int16)
    print('preprocess time: ', time3 - time2)
    # the order of SimpleITK is zyx, nibabel is xyz. ana_det use nibabel, so we need to reverse the order
    ori_shape = ori_shape[::-1]
    # then resize the corner coordinates to the corresponding coordinates in the resized image
    for key in corner_cor_dic.keys():
        try:
            pred_key_array = np.zeros_like(gt_data, dtype=np.int16)
            ngt_data = np.zeros_like(gt_data)
            ngt_data[gt_data==key] = 1
            sam_segs = {}
            sam_bboxes = {}
            sam_slice_dice_scores = {}
            ##### locate and segment each part of the organ #####
            img_id_ls = []
            # transfer the corner coordinates to the resized image
            corner_cor_dic[key][0] = [[corner_cor_dic[key][0][0][0], np.around(corner_cor_dic[key][0][0][1]*1024/ori_shape[1]), 
                                    np.around(corner_cor_dic[key][0][0][2]*1024/ori_shape[2])],
                                    [corner_cor_dic[key][0][1][0], np.around(corner_cor_dic[key][0][1][1]*1024/ori_shape[1]), 
                                    np.around(corner_cor_dic[key][0][1][2]*1024/ori_shape[2])]
                                    ]
            padding = 10
            x_min = corner_cor_dic[key][0][0][-1]-padding
            x_max = corner_cor_dic[key][0][1][-1]+padding
            y_min = corner_cor_dic[key][0][0][-2]-padding
            y_max = corner_cor_dic[key][0][1][-2]+padding
            z_min = corner_cor_dic[key][0][0][0]
            z_max = corner_cor_dic[key][0][1][0]
            if len(imgs.keys())>0:
                for img_id in imgs.keys():
                    if img_id>=z_min and img_id<=z_max:
                        # get bounding box from mask
                        gt2D = gts[img_id]
                        ngt2D = np.zeros_like(gt2D)
                        ngt2D[gt2D==key] = 1
                        ori_img = imgs[img_id]
                        bbox = np.array([x_min, y_min, x_max, y_max])
                        seg_mask, seg_prob = finetune_model_predict(ori_img, bbox, sam_trans, sam_model_tune)
                        sam_segs[img_id] = seg_mask
                        sam_bboxes[img_id] = bbox
                        img_id_ls.append(img_id)
                        pred_key_array[img_id] = (cv2.resize(seg_prob, pred_key_array[img_id].shape, interpolation=cv2.INTER_NEAREST)> 0.5).astype(np.uint8)
                        slice_dice, slice_intersect, slice_volume = compute_dice(seg_mask>0, ngt2D>0)
                        sam_slice_dice_scores[img_id] = slice_dice
                if len(img_id_ls) > 0:
                    # visualize segmentation results
                    img_id = random.choice(img_id_ls)
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(imgs[img_id])
                    show_box(sam_bboxes[img_id], axes[0])
                    show_mask(1*(gts[img_id]==key), axes[0])
                    axes[0].set_title('Ground Truth')
                    axes[0].axis('off')

                    axes[1].imshow(imgs[img_id])
                    show_box(sam_bboxes[img_id], axes[1])
                    show_mask(sam_segs[img_id], axes[1])
                    axes[1].set_title('DSC={:.3f}'.format(sam_slice_dice_scores[img_id]))
                    axes[1].axis('off')
                    fig.savefig(join(config_file['data']['seg_png_save_path'], '{0}_{1}_{2}_cl{3}.png'.format(os.path.basename(args.config_file).replace('test_','').replace('.txt',''), \
                                    nii_path.split('/')[-1].split('.')[0], str(id), str(key))))
                    plt.close(fig)
            
            # compute dice score for the whole volume
            volume_dice = 2*np.sum(ngt_data*pred_key_array)/(np.sum(1*ngt_data+pred_key_array)+0.0001)
            volume_hd95 = hd95(ngt_data, pred_key_array, voxelspacing=gt_sitk.GetSpacing()[::-1])
            print('class:', key, 'volume dice score: ', volume_dice, 'volume hd95: ', volume_hd95)
            sam_dice_scores[key].append(volume_dice)
            sam_hd95_scores[key].append(volume_hd95)

            # save nii
            pred_sitk = sitk.GetImageFromArray(pred_key_array)
            pred_sitk.SetSpacing(gt_sitk.GetSpacing())
            pred_sitk.SetOrigin(gt_sitk.GetOrigin())
            pred_sitk.SetDirection(gt_sitk.GetDirection())
            sitk.WriteImage(pred_sitk, join(config_file['data']['seg_save_path'], '{0}_{1}_{2}_cl{3}.nii.gz'.format(os.path.basename(args.config_file).replace('test_','').replace('.txt',''), \
                                        nii_path.split('/')[-1].split('.')[0], str(id), str(key))))
        except Exception:
            traceback.print_exc()
            print('error in {0}, class {1}'.format(nii_path, key))

    time4 = time.time()
    print('segment time: ', time4 - time3)
    print('total time: ', time4 - time1)

for key in config_file['data']['fg_class']:
    print('DSC for class {}: {:.3f}'.format(key, np.mean(sam_dice_scores[key])), 'HD95 for class {}: {:.3f}'.format(key, np.mean(sam_hd95_scores[key])))

# save all the result as JSON
with open(join(f'{current_directory}/result/json', '{0:}').format(os.path.basename(args.config_file).replace('test_','dsc_')), 'w') as f:
    json.dump(sam_dice_scores, f, cls=NumpyEncoder)

with open(join(f'{current_directory}/result/json', '{0:}').format(os.path.basename(args.config_file).replace('test_','hd95_')), 'w') as f:
    json.dump(sam_hd95_scores, f, cls=NumpyEncoder)

#% save mean+-std dice scores as txt
with open(join(f'{current_directory}/result/dsc', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_dice_scores[key] is a list or numpy array
        if isinstance(sam_dice_scores[key], (list, np.ndarray)) and len(sam_dice_scores[key]) > 0:
            mean_score = np.mean(sam_dice_scores[key])
            std_score = np.std(sam_dice_scores[key])
            f.write('DSC for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_score, std_score))
        else:
            print("sam_dice_scores[{}] is not a list or numpy array or it is empty".format(key))

#% save mean+-std hd95 scores as txt
with open(join(f'{current_directory}/result/hd95', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_hd95_scores[key] is a list or numpy array
        if isinstance(sam_hd95_scores[key], (list, np.ndarray)) and len(sam_hd95_scores[key]) > 0:
            mean_score = np.mean(sam_hd95_scores[key])
            std_score = np.std(sam_hd95_scores[key])
            f.write('HD95 for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_score, std_score))
        else:
            print("sam_hd95_scores[{}] is not a list or numpy array or it is empty".format(key))