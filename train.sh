python train.py \
    --config configs/default.yaml \
    --data_path /path/to/your/data/ksp_full.h5 \
    --output_dir outputs/acc8_pts500_long_split \
    --acceleration 4 \
    --initial_points 500 \
    --max_iterations 2000 \
    --gpu 2