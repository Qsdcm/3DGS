bash scripts/test.sh \
    --dataset "/data/data54/wanghaobo/data/ksp_full.h5" \
    --weights "/data/data54/wanghaobo/3DGS/3dgsVC/outputs/checkpoints/best.pth" \
    --acceleration 8 \
    --slices_axial "50 100 150" \
    --slices_coronal "60 120" \
    --slices_sagittal "70 140"