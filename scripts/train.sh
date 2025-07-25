PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -u ./tools/train.py \
    --name with_cm \
    --batch_size 32 \
    --num_epochs 5000 \
    --dataset_name both57M
