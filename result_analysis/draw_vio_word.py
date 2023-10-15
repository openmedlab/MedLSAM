import json
import matplotlib.pyplot as plt
import numpy as np

# Load the file content
with open("result/json/word_medlam.txt", "r") as file:
    data = json.load(file)

# Adjusting the function to parse the nested list structure

def compute_3d_iou_v2(pred, gt):
    # Extract corners for predicted and ground truth
    pred_x1, pred_y1, pred_z1 = pred[0]
    pred_x2, pred_y2, pred_z2 = pred[1]
    
    gt_x1, gt_y1, gt_z1 = gt[0]
    gt_x2, gt_y2, gt_z2 = gt[1]
    
    # Compute the volume of intersection
    x_overlap = max(0, min(pred_x2, gt_x2) - max(pred_x1, gt_x1))
    y_overlap = max(0, min(pred_y2, gt_y2) - max(pred_y1, gt_y1))
    z_overlap = max(0, min(pred_z2, gt_z2) - max(pred_z1, gt_z1))
    intersection_volume = x_overlap * y_overlap * z_overlap
    
    # Compute the volume of the two boxes
    pred_volume = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) * (pred_z2 - pred_z1)
    gt_volume = (gt_x2 - gt_x1) * (gt_y2 - gt_y1) * (gt_z2 - gt_z1)
    
    # Compute IoU
    iou = intersection_volume / (pred_volume + gt_volume - intersection_volume)
    
    return iou

# Compute IoU for each entry and category using the adjusted function
iou_results_v2 = {}
for key, value_list in data.items():
    iou_list = []
    for item in value_list:
        pred_corner, gt_corner, _ = item
        iou_list.append(compute_3d_iou_v2(pred_corner[0], gt_corner))
    iou_results_v2[key] = iou_list

plt.figure(figsize=(20, 5))

# Extracting categories and their respective IoU values
categories = list(iou_results_v2.keys())
# fg_categories = ['Brain Stem','Eye L', 'Eye R','Lens L','Lens R','Opt Nerve L','Opt Nerve R','Opt Chiasma','Temporal Lobes L','Temporal Lobes R',
#                 'Pituitary','Parotid Gland L','Parotid Gland R', 'Inner Ear L','Inner Ear R', 'Mid Ear L', 'Mid Ear R', 'TM Joint L', 'TM Joint R',
#                 'Spinal Cor', 'Mandible L', 'Mandible R']

fg_categories = ['Liver','Spleen','Kidney L','Kidney R','Stomach','Gallbladder','Esophagus','Pancreas','Duodenum','Colon',
                'Intestine','Adrenal','Rectum ','Bladder ','Head of Femur L','Head of Femur R']
iou_values = [iou_results_v2[cat] for cat in categories]

# Plotting the violin plot
plt.violinplot(iou_values, showmedians=True)
plt.xticks(range(1, len(categories) + 1), fg_categories, fontsize=12, rotation=45)
plt.title('IoU Distribution for Different Categories')
plt.xlabel('Category', fontsize=12)
plt.ylabel('IoU', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
plt.savefig('fig/word_medlam_iou.png')
plt.close()

# Define the wall distance function
def wd(box1, box2, spacing):
    'wall distance, spacing=[h,w,d]'
    box1 = np.asarray(box1).reshape([2, -1])
    box2 = np.asarray(box2).reshape([2, -1])
    spacing = np.asarray(spacing).reshape([1, -1])
    wall_distance = np.mean(np.abs(box1 - box2) * spacing, axis=0)
    return wall_distance

# Compute wall distance for each entry and dimension
wd_results = {dim: {cat: [] for cat in data.keys()} for dim in ['h', 'w', 'd']}

for key, value_list in data.items():
    for item in value_list:
        pred_corner, gt_corner, spacing = item
        distances = wd(pred_corner[0], gt_corner, spacing)
        for dim_idx, dim in enumerate(['h', 'w', 'd']):
            wd_results[dim][key].append(distances[dim_idx])

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(20, 15))

for idx, (dim, ax) in enumerate(zip(['h', 'w', 'd'], axes)):
    iou_dim_values = [wd_results[dim][cat] for cat in categories]
    ax.violinplot(iou_dim_values, showmedians=True)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(fg_categories, rotation=45)
    ax.set_title(f'Absolute Wall Distance in {dim.upper()} Dimension for Different Categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('Wall Distance')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
plt.savefig('fig/word_medlam_wd.png')
plt.close()