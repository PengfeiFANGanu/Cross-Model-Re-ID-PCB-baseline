#SYSU dataset
#ALL SEARCH MODE
# python train.py --dataset sysu --lr 0.1 --method agw --gpu 0

python train_PCB_1.py --dataset sysu --lr 0.1 --method agw --gpu 6

python test.py --mode all --resume './save_model/sysu_agw_p4_n4_lr_0.1_seed_0_best.t' --gpu 7 --dataset sysu


python train_PCB_1_ctn.py --dataset sysu --lr 0.1 --method agw --gpu 7

python test.py --mode all --resume './save_model/sysu_agw_p4_n4_lr_0.1_seed_0_mode_all_alpha_1.0_best.t' --gpu 7 --dataset sysu
#INDOOR SEARCH MODE

python train_PCB_1.py --dataset sysu --lr 0.1 --method agw --gpu 6 --mode indoor

python test.py --resume './save_model/sysu_agw_p4_n4_lr_0.1_seed_0_best.t' --gpu 6 --dataset sysu --mode indoor



python train_PCB_1_ctn.py --dataset sysu --lr 0.1 --method agw --gpu 7 --alpha 1.0 --mode indoor

python test.py --resume './save_model/sysu_agw_p4_n4_lr_0.1_seed_0_mode_indoor_alpha_1.0_best.t' --gpu 7 --dataset sysu --mode indoor







#REGDB dataset
#VISIBLE TO INFRARED

python train_PCB_1.py --dataset regdb --lr 0.1 --method agw --gpu 4 --batch-size 4 --trial 1

python test.py --resume './save_model/regdb_agw_p4_n4_lr_0.1_seed_0_trial_modality_v2i_best.t' --gpu 4 --dataset regdb --modality v2i


python train_PCB_1_ctn.py --dataset regdb --lr 0.1 --method agw --gpu 5 --batch-size 4 --trial 1

python test.py --resume './save_model/regdb_agw_p4_n4_lr_0.1_seed_0_trial_1_modality_v2i_alpha_1.0_best.t' --gpu 5 --dataset regdb --modality v2i


#INFRARED TO VISIBLE
python train_PCB_1.py --dataset regdb --lr 0.1 --method agw --gpu 4 --batch-size 4 --trial 1 --modality i2v
python test.py --resume './save_model/regdb_agw_p4_n4_lr_0.1_seed_0_trial_modality_v2i_best.t' --gpu 4 --dataset regdb --modality i2v


python train_PCB_1_ctn.py --dataset regdb --lr 0.1 --method agw --gpu 5 --batch-size 4 --trial 1 --modality i2v
python test.py --resume './save_model/regdb_agw_p4_n4_lr_0.1_seed_0_trial_1_modality_v2i_alpha_1.0_best.t' --gpu 5 --dataset regdb --modality i2v