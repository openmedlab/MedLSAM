# %% load environment
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
import torch
from MedSAM.segment_anything.build_sam import sam_model_registry
from MedSAM.segment_anything.utils.transforms import ResizeLongestSide
from tqdm import trange
import argparse
import traceback
from MedSAM.auto_pre_CT import preprocess_ct

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
    return medsam_seg

#% run inference
# set up the parser
parser = argparse.ArgumentParser(description='run inference on testing set based on MedSAM')
parser.add_argument('-c', '--config_file', type=str, default='config/test_config/test.txt', help='path to the config file')
args = parser.parse_args()
config_file = parse_config(args.config_file)

#% load MedSAM model
sam_model_tune = sam_model_registry[config_file['vit']['net_type']](checkpoint=config_file['weight']['vit_load_path']).to('cuda:0')
sam_trans = ResizeLongestSide(sam_model_tune.image_encoder.img_size)

nii_pathes = read_file_list(config_file['data']['query_image_ls'])
gt_pathes = read_file_list(config_file['data']['query_label_ls'])
os.makedirs(config_file['data']['seg_png_save_path'], exist_ok=True)
os.makedirs('result/dsc', exist_ok=True)

sam_dice_scores = {key:[] for key in config_file['data']['fg_class']}

for id in trange(len(nii_pathes)):
    nii_path = nii_pathes[id]
    gt_path = gt_pathes[id]
    save_path = join(config_file['data']['seg_save_path'], nii_path.split('/')[-1].split('.')[0] + '.npz')
    # then resize the corner coordinates to the corresponding coordinates in the resized image
    for key in config_file['data']['fg_class']:
        try:
            imgs, gts = preprocess_ct(gt_path, nii_path, label_id=key, image_size=1024, gt_slice_threshold=config_file['data']['gt_slice_threshold'])

            sam_segs = []
            sam_bboxes = []
            sam_slice_dice_scores = []
            volume_intersect = 0
            volume_sum = 0.001

            for img_id, ori_img in enumerate(imgs):
                # get bounding box from mask
                gt2D = gts[img_id]
                y_indices, x_indices = np.where(gt2D > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = gt2D.shape
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                bbox = np.array([x_min, y_min, x_max, y_max])
                seg_mask = finetune_model_predict(ori_img, bbox, sam_trans, sam_model_tune)
                sam_segs.append(seg_mask)
                sam_bboxes.append(bbox)
                # these 2D dice scores are for debugging purpose. 
                # 3D dice scores should be computed for 3D images
                slice_dice, slice_intersect, slice_volume = compute_dice(seg_mask>0, gt2D>0)
                volume_intersect += slice_intersect
                volume_sum += slice_volume
                sam_slice_dice_scores.append(slice_dice)
            volume_dice = 2*volume_intersect/volume_sum
            sam_dice_scores[key].append(volume_dice)
            
            # save nii, including sam_segs, sam_bboxes, sam_dice_scores
            np.savez_compressed(save_path, medsam_segs=sam_segs, gts=gts, sam_bboxes=sam_bboxes)

            # visualize segmentation results
            img_id = np.random.randint(0, len(imgs))
            # show ground truth and segmentation results in two subplots
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(imgs[img_id])
            show_box(sam_bboxes[img_id], axes[0])
            show_mask(gts[img_id], axes[0])
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            axes[1].imshow(imgs[img_id])
            show_box(sam_bboxes[img_id], axes[1])
            show_mask(sam_segs[img_id], axes[1])
            axes[1].set_title('DSC={:.3f}'.format(sam_slice_dice_scores[img_id]))
            axes[1].axis('off')
            # save figure
            fig.savefig(join(config_file['data']['seg_png_save_path'], '{0}_{1}_cl{2}.png'.format(nii_path.split('/')[-1].split('.')[0], str(id), str(key))))
            # close figure
            plt.close(fig)
        except Exception:
            traceback.print_exc()
            print('error in {0}, class {1}'.format(nii_path, key))

#% save dice scores
for key in config_file['data']['fg_class']:
    print('DSC for class {}: {:.3f}'.format(key, np.mean(sam_dice_scores[key])))

#% save dice scores as txt
with open(join('result/dsc', '{0:}').format(os.path.basename(args.config_file).replace('test_','')), 'w') as f:
    for key in config_file['data']['fg_class']:
        # Check if sam_dice_scores[key] is a list or numpy array
        if isinstance(sam_dice_scores[key], (list, np.ndarray)) and len(sam_dice_scores[key]) > 0:
            mean_score = np.mean(sam_dice_scores[key])
            std_score = np.std(sam_dice_scores[key])
            f.write('DSC for class {}: {:.3f} +- {:.3f}\n'.format(key, mean_score, std_score))
        else:
            print("sam_dice_scores[{}] is not a list or numpy array or it is empty".format(key))