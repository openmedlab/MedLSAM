import os

# 指定的目录路径
directory_ls = ["/mnt/data/oss_beijing/Data/FLARE2023/validation_normal_gt/"]

total_files = []
for directory in directory_ls:
    # 找到所有以 'total.nii.gz' 结尾的文件
    total_files += [os.path.join(root, name)
                    for root, dirs, files in os.walk(directory)
                    for name in files
                    if name.endswith(".nii.gz")]

# 将这些文件路径写入一个txt文件
with open("/mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_data/FLARE23/label.txt", "w") as f:
    for file_path in total_files:
        f.write(file_path + "\n")

# 替换 "_total.nii.gz" 为 ".nii.gz"
replaced_files = [file_path.replace("validation_normal_gt", "validation") for file_path in total_files]

# 将替换后的文件路径写入另一个txt文件
with open("/mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_data/FLARE23/image.txt", "w") as f:
    for file_path in replaced_files:
        f.write(file_path + "\n")
