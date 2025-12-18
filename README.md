## 3DGSMR (Paper Reproduction)

This folder contains a self-contained prototype that follows the *Three-Dimensional MRI Reconstruction with Gaussian Representations* paper. It loads k-space data from the provided HDF5 file, builds a 3D Gaussian representation of the MR volume, and optimizes it with the long-axis splitting strategy and k-space data-consistency losses described in the manuscript.

### Project Layout

- `train.py` – entry point that prepares the mask, initializes the Gaussian field, runs optimization, and writes checkpoints.
- `gsmr3d/` – package with reusable modules:
  - `data.py` – HDF5 loading helpers plus FFT/iFFT wrappers.
  - `masks.py` – variable-density Gaussian sampling masks for k-space.
  - `gaussian_model.py` – Gaussian representation, voxelizer, and adaptive densification (long-axis splitting, pruning).
  - `losses.py` – magnitude TV loss from the paper.
  - `metrics.py` – 3D PSNR and SSIM utilities for progress monitoring.
- `requirements.txt` – minimal dependency list (`torch`, `numpy`, `h5py`).

### Usage

1. Activate an environment with PyTorch, NumPy, and h5py.
2. Run training from the current folder, pointing to your single HDF5 file:

```bash
python train.py \
  --data /data/data54/wanghaobo/data/ksp_full.h5 \
  --acceleration 4 \
  --mask-seed 2025 \
  --device cuda:0 \
  --num-gaussians 20000 \
  --max-iters 600 \
  --save-dir outputs/af4_run
```

Key arguments:

- `--device` lets you pick a GPU ID (e.g., `cuda:1`) or CPU.
- `--acceleration`, `--center-fraction`, and `--mask-seed` control the Gaussian undersampling mask.
- `--num-gaussians`, `--max-gaussians`, and `--densify-interval` configure initialization and adaptive long-axis splitting.
- `--tv-weight` tunes the magnitude TV regularization.
- `--checkpoint-every` makes the script periodically save intermediate reconstructions.

Outputs (mask, best reconstruction) are written to `--save-dir` as PyTorch tensors and can be visualized later.

### Notes and Assumptions

- The provided dataset contains fully-sampled multi-coil data stored as a complex-valued compound dataset. To keep the prototype lightweight, the current pipeline combines coils by summation before reconstruction; more advanced coil sensitivity estimation can be inserted inside `gsmr3d/data.py` if needed.
- The voxelizer follows the 3-sigma rule and the long-axis splitting described in the paper. Each Gaussian influences only a local patch, making the method practical on a single GPU while retaining differentiability for all parameters.
- Training defaults mirror the paper (600 iterations, densification every 100 steps, maximum 400k Gaussians). You can lower `--num-gaussians`/`--max-gaussians` for debugging or increase them for higher fidelity when resources permit.

This implementation is designed for code bring-up. Once the pipeline runs end-to-end on the provided HDF5 file, you can iterate on hyper-parameters, implement more precise coil modeling, or plug in alternate masks without changing the training skeleton.
