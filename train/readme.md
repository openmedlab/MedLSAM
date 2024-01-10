## Training
### Training Data preparation
- Create a `train/config/ori_nii.txt` file listing the paths to the original CT nii files.(MedLAM is based on the self-supervised learning tasks and no label file is required during the training time!!!)
- run `python train/dataset_preprocess.py`. It will automatically preprocess the CT file. By default, the preprocessed CT files will be saved with a new name that appends `_pre` to the original filename. For example, if your original file is named `scan.nii`, the preprocessed file will be named `scan_pre.nii`.
- After preprocessing, the paths to the preprocessed CT files will be automatically saved in a file named `pre_nii.txt` located in the `train/config/` directory.
### Training script
```bash 
python train/train_position_full_size_with_fc.py -c train/config/train_position_full_size_with_fc.txt
```
- the checkpoint will be saved in `train/checkpoint`

