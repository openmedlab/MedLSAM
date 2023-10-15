import os
import sys

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

import matplotlib.pyplot as plt
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))

join = os.path.join
import argparse
import json
import traceback

from tqdm import trange

from data_process.data_process_func import *
from MedLAM.Anatomy_detection import AnatomyDetection
from MedLAM.detection_functions import *
from MedSAM.auto_pre_CT import *
from util.parse_config import parse_config

# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedLAM')
parser.add_argument('-c', '--config_file', type=str, help='path to the config file')
args = parser.parse_args()
config_file = parse_config(args.config_file)

#% load MedSAM model
ana_det = AnatomyDetection(args.config_file)

nii_pathes = read_file_list(config_file['data']['query_image_ls'])
gt_pathes = read_file_list(config_file['data']['query_label_ls'])
lam_iou_dic = {}
lam_result_dic = {}
lam_wd_dic = {}
os.makedirs(f'{current_directory}/result/iou', exist_ok=True)
os.makedirs(f'{current_directory}/result/wd', exist_ok=True)


for id in trange(len(nii_pathes)):
    nii_path = nii_pathes[id]
    gt_path = gt_pathes[id]
    
    gt_data = nibabel.load(gt_path)
    gt_spacing = gt_data.header['pixdim'][1:4][::-1]
    print('gt_spacing:', gt_spacing)
    gt_mask = gt_data.get_fdata().transpose(2, 1, 0)
    extreme_cor_dic, corner_cor_dic, ori_shape = ana_det.get_extreme_corner(nii_path)
    for key in corner_cor_dic.keys():
        try:
            gt_corner = get_bound_coordinate(1*(gt_mask==int(key)))
            print('file', nii_path, '\n gt corner:', gt_corner, '\n predict corner:',corner_cor_dic[key])
            if key not in lam_iou_dic.keys():
                lam_result_dic[key] = [[corner_cor_dic[key], gt_corner, gt_spacing]]
                lam_iou_dic[key] = [iou(corner_cor_dic[key], gt_corner)]
                lam_wd_dic[key] = [np.mean(wd(corner_cor_dic[key], gt_corner, gt_spacing))]
            else:
                lam_result_dic[key].append([corner_cor_dic[key], gt_corner, gt_spacing])
                lam_iou_dic[key].append(iou(corner_cor_dic[key], gt_corner))
                lam_wd_dic[key].append(np.mean(wd(corner_cor_dic[key], gt_corner, gt_spacing)))
        except Exception:
            traceback.print_exc()
            print('error in {}'.format(nii_path))

# save the result as JSON
os.makedirs(f'{current_directory}/result/json', exist_ok=True)
with open(join(f'{current_directory}/result/json', '{0:}').format(os.path.basename(args.config_file).replace('test_','iou_')), 'w') as f:
    json.dump(lam_result_dic, f, cls=NumpyEncoder)


#% save iou as txt
with open(join(f'{current_directory}/result/iou', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_dice_scores[key] is a list or numpy array
        if isinstance(lam_iou_dic[key], (list, np.ndarray)) and len(lam_iou_dic[key]) > 0:
            mean_score = np.mean(lam_iou_dic[key])
            std_score = np.std(lam_iou_dic[key])
            f.write('MedSAM IOU for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_score, std_score))
        else:
            print("lam_iou_scores[{}] is not a list or numpy array or it is empty".format(key))

#% save wd as txt
with open(join(f'{current_directory}/result/wd', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_dice_scores[key] is a list or numpy array
        if isinstance(lam_wd_dic[key], (list, np.ndarray)) and len(lam_wd_dic[key]) > 0:
            mean_scores = np.mean(lam_wd_dic[key], axis=0)
            std_scores = np.std(lam_wd_dic[key], axis=0)
            f.write('MedSAM WD for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_scores, std_scores))
        else:
            print("lam_wd_scores[{}] is not a list or numpy array or it is empty".format(key))

