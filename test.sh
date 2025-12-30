bash scripts/test.sh \
    --dataset "/data0/congcong/data/3D_data/h5data/ksp_full.h5" \
    --weights "/data0/congcong/code/haobo/V3/3DGS/3dgsVC/outputs/acc8_pts500_grad0.005_seed42/checkpoints/best.pth" \
    --acceleration 8 \
    --slices_axial "50 100 150" \
    --slices_coronal "60 120" \
    --slices_sagittal "70 140"