PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1 python3 -u ../tools/generate_results.py checkpoints/both57M/with_cm/opt.txt 0