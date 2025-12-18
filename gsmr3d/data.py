"""MRI data I/O helpers."""

from __future__ import annotations

import pathlib
from typing import Tuple

import h5py
import numpy as np
import torch


def load_kspace_volume(
    path: str | pathlib.Path,
    *,
    combine_coils: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Load complex k-space data stored as a compound dataset."""

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5f:
        if "kspace" not in h5f:
            raise KeyError("Dataset 'kspace' not found")
        raw = h5f["kspace"][...]

    if np.iscomplexobj(raw) or raw.dtype.kind == "c":
        data = raw.astype(np.complex64, copy=False)
    elif raw.dtype.fields and "r" in raw.dtype.fields and "i" in raw.dtype.fields:
        real = np.asarray(raw["r"], dtype=np.float32)
        imag = np.asarray(raw["i"], dtype=np.float32)
        data = real + 1j * imag
    else:
        raise TypeError(f"Unsupported k-space dtype: {raw.dtype}")

    tensor = torch.from_numpy(data).to(dtype=dtype)
    if combine_coils:
        tensor = tensor.sum(dim=0)

    return tensor


def fft3c(volume: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 3D FFT."""

    return torch.fft.fftn(volume, dim=(-3, -2, -1), norm="ortho")


def ifft3c(kspace: torch.Tensor) -> torch.Tensor:
    """Inverse counterpart of :func:`fft3c`."""

    return torch.fft.ifftn(kspace, dim=(-3, -2, -1), norm="ortho")
