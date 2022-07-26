# DualNet-Lesion-Segmentation: Dual-Branch Network with Dual-Sampling Modulated Dice Loss for Hard Exudate Segmentation in Colour Fundus Images

This repository is the official PyTorch implementation of
paper: [Dual-Branch Network with Dual-Sampling Modulated Dice Loss for Hard Exudate Segmentation in Colour Fundus Images](https://ieeexplore.ieee.org/abstract/document/9525200)
.

## Code Structure

- The **dual-HED** folder contains vanilla and DualNet versions of HED.
- The **dual-segmentation-toolbox** folder contains vanilla and DualNet versions of PSPNet and Deeplabv3.
- The preprocessing and evaluation scripts can be found in **scripts** folder.

## Environment

- Python: 3.7
- PyTorch: 1.1.0

- The dual-segmentation-toolbox code also needs [apex](https://github.com/NVIDIA/apex)
  and [inplace-abn](https://github.com/mapillary/inplace_abn.git).

```sh
conda create -n dualnet python=3.7 -y
conda activate dualnet

pip install torch==1.1.0 torchvision==0.3.0

cd apex
python setup.py install --cpp_ext
pip install inplace_abn==1.0.12

pip install opencv-python
pip install tqdm scipy scikit-image

# for evaluation scripts
pip install pandas sklearn
pip install cupy-cuda100
```

The ImageNet pretrained weights of backbones are available
at this [link](https://drive.google.com/drive/folders/11IDaVAW8Dody0xDIJ-U-e_LDE3rnxEyk?usp=sharing).

## Training and Evaluation

Please refer to [train.sh](./train.sh) and [eval.sh](./eval.sh).

## Results and models

We evaluate our DualNet
on [DDR](https://github.com/nkicsl/DDR-dataset)
and [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) datasets.

### DDR

| Method             |  IoU  | F<sub>pixel</sub> | AUPR  |
|:-------------------|:-----:|:-----------------:|:-----:|
| Dual PSPNet+DSM    | 40.40 |       57.55       | 54.86 |
| Dual DeepLabV3+DSM | 38.62 |       55.72       | 52.29 |
| Dual HED+DSM       | 41.39 |       58.55       | 49.90 |

### IDRiD

| Method             |  IoU  | F<sub>pixel</sub> | AUPR  |
|:-------------------|:-----:|:-----------------:|:-----:|
| Dual PSPNet+DSM    | 61.03 |       75.80       | 77.82 |
| Dual DeepLabV3+DSM | 61.53 |       76.19       | 76.69 |
| Dual HED+DSM       | 61.16 |       75.90       | 79.05 |

The corresponding trained weights of our DualNet models are available at
this [link](https://drive.google.com/drive/folders/1meCLBR7FuK4pswe52uBxMzuBFHC31lW_?usp=sharing).

In the paper, we reported average performance over three repetitions, while our code only reported the best one among
them.

## Acknowledgements

This code is heavily borrowed from [HED](https://github.com/s9xie/hed)
, [pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox/tree/pytorch-1.1),
and [BBN](https://github.com/Megvii-Nanjing/BBN). Thanks for their contributions.

## Citation

If you find this code useful in your research, please consider citing:

```
@article{liu2022dual,
  title={Dual-Branch Network With Dual-Sampling Modulated Dice Loss for Hard Exudate Segmentation in Color Fundus Images},
  author={Liu, Qing and Liu, Haotian and Zhao, Yang and Liang, Yixiong},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={26},
  number={3},
  pages={1091--1102},
  year={2022},
  publisher={IEEE}，
  doi={10.1109/JBHI.2021.3108169}
}
```
