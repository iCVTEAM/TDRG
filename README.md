# TDRG
Pytorch implementation of [Transformer-based Dual Relation Graph for Multi-label Image Recognition. ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Transformer-Based_Dual_Relation_Graph_for_Multi-Label_Image_Recognition_ICCV_2021_paper.html)

![TDRG](https://github.com/iCVTEAM/TDRG/blob/master/figs/motivation.png)

## Prerequisites

Python 3.6+

Pytorch 1.6

CUDA 10.1

Tesla V100 × 2

## Datasets

- MS-COCO: [train](http://images.cocodataset.org/zips/train2014.zip)  [val](http://images.cocodataset.org/zips/val2014.zip)  [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- VOC 2007: [trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)  [test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)  [test_anno](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar)

## Train

```
CUDA_VISIBLE_DEVICES=0,1 python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 0.03 -b 64
```

## Test

```
python main.py  --data COCO2014 --data_root_dir $DATA_PATH$ --save_dir $SAVE_PATH$ --i 448  --lr 0.03 -b 64 -e --resume checkpoint/COCO2014/checkpoint_COCO.pth
```

## Visualization

![vis](https://github.com/iCVTEAM/TDRG/blob/master/figs/vis.png)

## Citation

- If you find this work is helpful, please cite our paper

```
@InProceedings{Zhao2021TDRG,
    author    = {Zhao, Jiawei and Yan, Ke and Zhao, Yifan and Guo, Xiaowei and Huang, Feiyue and Li, Jia},
    title     = {Transformer-Based Dual Relation Graph for Multi-Label Image Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {163-172}
}
```

