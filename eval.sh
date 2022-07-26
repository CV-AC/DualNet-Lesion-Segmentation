# Testing for "dual-HED"
# e.g.
model='dual_hed'       dataset='IDRID'; python dual-HED/main.py --model $model --dataset $dataset --output '../output/'$model'_'$dataset --checkpoint '/path/to/'$model'_'$dataset'.pt' --test-only

# Testing for "dual-segmentation-toolbox"
# e.g.
model='dual_pspnet'    dataset='DDR'; python dual-segmentation-toolbox/test.py --model $model --dataset $dataset --output 'output/'$model'_'$dataset --checkpoint '/path/to/'$model'_'$dataset'.pt'
model='dual_deeplabv3' dataset='DDR'; python dual-segmentation-toolbox/test.py --model $model --dataset $dataset --output 'output/'$model'_'$dataset --checkpoint '/path/to/'$model'_'$dataset'.pt'



# Evaluation
# e.g.
model='dual_pspnet' dataset='IDRID'
python scripts/eval.py        --dataset $dataset --pred 'output/'$model'_'$dataset
python scripts/eval_aupr.py   --dataset $dataset --pred 'output/'$model'_'$dataset
python scripts/eval_region.py --dataset $dataset --pred 'output/'$model'_'$dataset
