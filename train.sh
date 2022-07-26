# Prepare datasets
python scripts/prepare_dataset.py



# Training on DDR dataset

# PSPNet (pspnet, dual_pspnet)
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model pspnet --dataset 'DDR' --epochs 100 --output 'output/pspnet_DDR'
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model dual_pspnet --dataset 'DDR' --epochs 100 --dual-p 0.5 --output 'output/dual_pspnet_DDR'

# DeepLabV3 (deeplabv3, dual_deeplabv3)
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model deeplabv3 --dataset 'DDR' --epochs 100 --output 'output/deeplabv3_DDR'
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model dual_deeplabv3 --dual-p 0.25 --dataset 'DDR' --epochs 100 --output 'output/dual_deeplabv3_DDR'

# HED (hed, dual_hed)
python dual-HED/main.py --model hed --dataset 'DDR' --epochs 100 --output '../output/hed_DDR'
python dual-HED/main.py --model dual_hed --dual-p 0.25 --dataset 'DDR' --epochs 100 --output '../output/dual_hed_DDR'



# Training on IDRiD dataset

# PSPNet (pspnet, dual_pspnet)
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model pspnet --dataset 'IDRID' --epochs 40 --output 'output/pspnet_IDRID'
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model dual_pspnet --dual-p 0.5 --dataset 'IDRID' --epochs 40 --output 'output/dual_pspnet_IDRID'

# DeepLabV3 (deeplabv3, dual_deeplabv3)
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model deeplabv3 --dataset 'IDRID' --epochs 40 --output 'output/deeplabv3_IDRID'
python -m torch.distributed.launch --nproc_per_node=2 dual-segmentation-toolbox/train_dist.py --model dual_deeplabv3 --dual-p 0.25 --dataset 'IDRID' --epochs 40 --output 'output/dual_deeplabv3_IDRID'

# HED (hed, dual_hed)
python dual-HED/main.py --model hed --dataset 'IDRID' --epochs 40 --output '../output/hed_IDRID'
python dual-HED/main.py --model dual_hed --dual-p 0.25 --dataset 'IDRID' --epochs 40 --output '../output/dual_hed_IDRID'
