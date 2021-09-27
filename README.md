# TDRG
Transformer-based Dual Relation Graph for Multi-label Image Recognition. ICCV 2021

### Train
CUDA_VISIBLE_DEVICES=0,1 python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 0.03 -b 64

### Test
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 0.03 -b 64 -e --resume checkpoint/COCO2014/checkpoint_COCO.pth

### Dataset
MS-COCO 2014 
VOC 2007

