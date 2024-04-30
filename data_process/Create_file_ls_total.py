import os
import json

# 打开json，提取key保存
with open('/mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/data_process/file_to_organs.json', 'r') as f:
    dile_dic = json.load(f)
    
file_ls = [key  for key in dile_dic.keys()]

# save as txt
with open('/mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_data/Totalseg/all_label.txt', 'w') as f:
    for item in file_ls:
        f.write("%s\n" % item)
