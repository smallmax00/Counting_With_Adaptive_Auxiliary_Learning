# Counting_With_Adaptive_Auxiliary_Learning


## Data Prepare
Prepare you own data, please refer to [C-3-FrameWork](https://github.com/gjy3035/C-3-Framework) for detailed instructions.

## SHA

- Download [checkpoint_best.pth](https://drive.google.com/file/d/1HaRTgBhW1Evr5NBOCduaDY2h2Xdkb4l5/view?usp=sharing) and put it into ./checkpoints/GCN_paper_SHAB/  

### prediction
```
python test_SHA.py  
```

## JHU
- Download [checkpoint_best.pth](https://drive.google.com/file/d/1qn5vWfWJFk97EYflbFBhb61KxMw6R6aw/view?usp=sharing) and put it into ./checkpoints/GCN_paper_JHU/  

### prediction
```
python test_JHU.py
```


## QNRF
- Download [checkpoint_best.pth](https://drive.google.com/file/d/1Lkxwr4MEcug2IxnzZWuwKQOvMbEZ0SdI/view?usp=sharing) and put it into ./checkpoints/GCN_paper_QNRF/  

### prediction
```
python test_QNRF.py
```



## NWPU-Crowd

**Our method achieved 76.4 MAE and 327.4 MSE on [NWPU-Crowd counting benchmark](https://www.crowdbenchmark.com/index.html)**


# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@article{Meng_2022_Adaptive_Counting,
    author    = {Meng, Yanda and Bridge, Joshua and Zhao, Yitian and Joddrell, Martha and Qiao, Yihong and Yang, Xiaoyun and Huang, Xiaowei and Zheng, Yalin},
    title     = {Transportation Object Counting with Graph-Based Adaptive Auxiliary Learning},
    journal = {IEEE Transactions on Intelligent Transportation System},
    year      = {2022},
}

```
