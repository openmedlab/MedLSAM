import sys
import os
sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from util.parse_config import parse_config
from data_process.data_process_func import *
from tqdm import trange
import argparse
import traceback
from MedSAM.auto_pre_CT import *
from MedLAM.Anatomy_detection import AnatomyDetection
from MedLAM.detection_functions import *

# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedLAM')
parser.add_argument('-c', '--config_file', type=str, default='config/test_config/test_structseg.txt', help='path to the config file')
args = parser.parse_args()
config_file = parse_config(args.config_file)

#% load MedSAM model
ana_det = AnatomyDetection(args.config_file)

nii_pathes = read_file_list(config_file['data']['query_image_ls'])
gt_pathes = read_file_list(config_file['data']['query_label_ls'])
lam_iou_dic = {}

for id in trange(len(nii_pathes)):
    nii_path = nii_pathes[id]
    gt_path = gt_pathes[id]
    
    gt_mask = nibabel.load(gt_path).get_fdata().transpose(2, 1, 0)
    extreme_cor_dic, corner_cor_dic, ori_shape = ana_det.get_extreme_corner(nii_path)
    for key in corner_cor_dic.keys():
        try:
            gt_corner = get_bound_coordinate(1*(gt_mask==int(key)))
            print('file', nii_path, 'gt corner:', gt_corner, 'predict corner:',corner_cor_dic[key])
            if key not in lam_iou_dic.keys():
                lam_iou_dic[key] = [iou(corner_cor_dic[key], gt_corner)]
            else:
                lam_iou_dic[key].append(iou(corner_cor_dic[key], gt_corner))
        except Exception:
            traceback.print_exc()
            print('error in {}'.format(nii_path))

#% save iou as txt
with open(join('result/iou', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_dice_scores[key] is a list or numpy array
        if isinstance(lam_iou_dic[key], (list, np.ndarray)) and len(lam_iou_dic[key]) > 0:
            mean_score = np.mean(lam_iou_dic[key])
            std_score = np.std(lam_iou_dic[key])
            f.write('MedSAM IOU for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_score, std_score))
        else:
            print("lam_iou_scores[{}] is not a list or numpy array or it is empty".format(key))
