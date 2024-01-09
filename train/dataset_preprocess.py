import os
import sys

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
join = os.path.join
from tqdm import trange
from data_process.data_process_func import *


### rescale CT image to 3*3*3 mm & normalize the HU from [-1000, -200, 200, 1500] to [0,0.2,0.8,1]
### this Segmental Linear Normalization Function was proposed in "Automatic Segmentation of Organs-at-Risk from Head-and-Neck CT using Separable Convolutional Neural Network with Hard-Region-Weighted Loss"


ori_file_ls_path = 'train/config/ori_nii.txt' # change it to the txt file which contains all your CT nii data path
ori_file_ls = read_file_list(ori_file_ls_path)
preprocess_file_ls = []

for ori_file_path in trange(ori_file_ls):
    preprocessed_sample = load_and_pre_ct(ori_file_path, mode='image')
    preprocessed_image = preprocessed_sample['image']
    file_save_path = ori_file_path.replace('.nii', '_pre.nii') # change it to file save path
    save_array_as_nifty_volume(preprocessed_image, pixel_spacing=[3,3,3])
    preprocess_file_ls.append(file_save_path)


with open('train/config/pre_nii.txt', 'w') as file:
    # 遍历列表中的每个元素
    for item in preprocess_file_ls:
        # 将每个元素写入文件，后面加上换行符
        file.write(item + '\n')

