PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u ../tools/visualization.py \
    --opt_path checkpoints/both57M/with_cm/opt.txt \
    --text "the left hand circles thuùb twice, other fingers remain natural, while the right hand makes a fist." \
    --motion_length 150 \
    --fname "generated_motion.npy" \
    --gpu_id 0
