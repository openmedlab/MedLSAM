CUDA_VISIBLE_DEVICES=0 python /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/MedSAM_Inference_total.py -c /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_config/Totalseg/test_totalseg_sam_0.txt &
CUDA_VISIBLE_DEVICES=1 python /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/MedSAM_Inference_total.py -c /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_config/Totalseg/test_totalseg_sam_1.txt &
CUDA_VISIBLE_DEVICES=2 python /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/MedSAM_Inference_total.py -c /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_config/Totalseg/test_totalseg_sam_2.txt &
CUDA_VISIBLE_DEVICES=3 python /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/MedSAM_Inference_total.py -c /mnt/data/smart_health_02/leiwenhui/Code/MedLSAM/config/my_test_config/Totalseg/test_totalseg_sam_3.txt 