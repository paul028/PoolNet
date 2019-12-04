#! /bin/bash

python joint_main.py --arch resnet --train_root C:/Users/paulvincentnonat/Documents/GitHub/Saliency_Dataset/DUTS/DUTS-TR --train_list C:/Users/paulvincentnonat/Documents/GitHub/Saliency_Dataset/DUTS/DUTS-TR/train_pair.lst --train_edge_root C:/Users/paulvincentnonat/Documents/GitHub/Saliency_Dataset/HED-BSDS_PASCAL --train_edge_list C:/Users/paulvincentnonat/Documents/GitHub/Saliency_Dataset/HED-BSDS_PASCAL/bsds_pascal_train_pair_r_val_r_small.lst
# you can optionly change the -lr and -wd params
