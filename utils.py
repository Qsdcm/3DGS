import os
import torch
import numpy as np
import argparse
import torch.fft as FFT
import glob
import scipy.io as scio
#import tensorflow as tf
import logging
sqrt = np.sqrt
import torch.nn.functional as F
import torchvision.transforms as T
# import sigpy as sp
from icecream import ic
from tqdm import tqdm
from scipy.linalg import null_space, svd
from optimal_thresh import optht
import sigpy as sp
import sigpy.mri.app as MR
from torch.utils.dlpack import to_dlpack, from_dlpack
from cupy import from_dlpack as cu_from_dlpack


class Aclass_sense:
    def __init__(self, csm, mask, lam):
        self.s = csm
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        Ax = sense(self.s, ksp)
        AHAx = adjsense(self.s, Ax)
        return AHAx

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res


def sense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny 
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    # m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm),1) 
    m = Emat_xyt(c2r(ksp), True, c2r(csm), 1)
    res = Emat_xyt(m, False, c2r(csm), 1)   
    # res  = fft2c_2d(m.unsqueeze(1) * csm)
    return r2c(res) - ksp

def adjsense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny 
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    # m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm),1)    
    # res  = fft2c_2d(m.unsqueeze(1) * csm)
    m = Emat_xyt(c2r(ksp), True, c2r(csm), 1)
    res = Emat_xyt(m, False, c2r(csm), 1) 
    return r2c(res) - ksp


class ConjGrad:
    def __init__(self, A, rhs, max_iter=5, eps=1e-10):
        self.A = A
        self.b = rhs
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, x):
        x = CG(x, self.b, self.A, max_iter=self.max_iter, eps=self.eps)
        return x
    

def dot_batch(x1, x2):
    batch = x1.shape[0]
    res = torch.reshape(x1 * x2, (batch, -1))
    # res = torch.reshape(x1 * x2, (-1, 1))
    return torch.sum(res, 1)


def CG(x, b, A, max_iter, eps):
    r = b - A.A(x)
    p = r
    rTr = dot_batch(torch.conj(r), r)
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    num_iter = 0
    for iter in range(max_iter):
        if rTr.abs().max() < eps:
            break
        Ap = A.A(p)
        alpha = rTr / dot_batch(torch.conj(p), Ap)
        alpha = torch.reshape(alpha, reshape)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = dot_batch(torch.conj(r), r)
        beta = rTrNew / rTr
        beta = torch.reshape(beta, reshape)
        p = r + beta * p
        rTr = rTrNew

        num_iter += 1
    return x


def cgSENSE(ksp, csm, mask, x0, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x0 = Emat_xyt(x0, False, c2r(csm), 1)
    x = cg_iter.forward(x=r2c(x0))
    x = x * (1 - mask) + ksp
    res = Emat_xyt(c2r(x), True, c2r(csm), 1)

    return res


def init_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed(seed)
    # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True  # 固定卷积算法, 设为True会导致卷积变慢
        torch.backends.cudnn.benchmark = False


def save_mat(save_dict, variable, file_name, index=0, Complex=True, normalize=True):
    # variable = variable.cpu().detach().numpy()
    if normalize:

        if Complex:
            variable = normalize_complex(variable)
        else:
            variable_abs = torch.abs(variable)
            coeff = torch.max(variable_abs)
            variable = variable / coeff
    variable = variable.cpu().detach().numpy()
    file = os.path.join(save_dict, str(file_name) +
                        '_' + str(index + 1) + '.mat')
    datadict = {str(file_name): np.squeeze(variable)}
    scio.savemat(file, datadict)


def hfssde_save_mat(config, variable, variable_name='recon', normalize=True):
    if normalize:
        variable = normalize_complex(variable)
    variable = variable.cpu().detach().numpy()
    save_dict = config.sampling.folder
    file_name = config.training.sde + '_acc' + config.sampling.acc + '_acs' + config.sampling.acs \
                    + '_epoch' + str(config.sampling.ckpt)
    file = os.path.join(save_dict, str(file_name) + '.mat')
    datadict = {variable_name: np.squeeze(variable)}
    scio.savemat(file, datadict)


def get_all_files(folder, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(folder, pattern))]
    return sorted(files)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def to_tensor(x):
    re = np.real(x)
    im = np.imag(x)
    x = np.concatenate([re, im], 1)
    del re, im
    return torch.from_numpy(x)


def crop(img, cropc, cropx, cropy):
    if img.ndim == 5:
        nb, nc, c, x, y = img.size()
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]
    elif img.ndim == 4:
        nb, c, x, y = img.size()
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
    return cimg

def crop_new(img, cropc, cropx, cropy):
    if img.ndim == 5:
        nb, nc, c, x, y = img.size()
        padc = max(0, cropc - c)
        padx = max(0, cropx - x)
        pady = max(0, cropy - y)

        # 先padding两边对称
        if padc > 0 or padx > 0 or pady > 0:
            padding = [  # 注意顺序是最后一维在前
                pady // 2, pady - pady // 2,
                padx // 2, padx - padx // 2,
                padc // 2, padc - padc // 2
            ]
            img = F.pad(img, padding, mode='constant', value=0)
            _, _, c, x, y = img.size()

        # 再中心裁剪
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startc:startc + cropc, startx:startx + cropx, starty:starty + cropy]

    elif img.ndim == 4:
        nb, c, x, y = img.size()
        padx = max(0, cropx - x)
        pady = max(0, cropy - y)

        if padx > 0 or pady > 0:
            padding = [
                pady // 2, pady - pady // 2,
                padx // 2, padx - padx // 2
            ]
            img = F.pad(img, padding, mode='constant', value=0)
            _, _, x, y = img.size()

        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startx:startx + cropx, starty:starty + cropy]

    else:
        raise ValueError(f"Unsupported tensor dimension: {img.ndim}")

    return cimg

def np_crop(img, cropc, cropx, cropy):
    if img.ndim == 5:
        nb, nc, c, x, y = img.shape
        startc = c // 2 - cropc // 2
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startc:startc + cropc, startx:startx + cropx, starty: starty + cropy]
    elif img.ndim == 4:
        nb, c, x, y = img.shape
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        cimg = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
    return cimg

def t_crop(img, cropx, cropy):

    nb, c, x, y = img.size()
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    cimg = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
    return cimg

def acs_crop(img, cropx, cropy):

    acs = torch.zeros_like(img)
    nb, c, x, y = img.size()
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    acs[:, :, startx:startx + cropx, starty: starty + cropy] = img[:, :, startx:startx + cropx, starty: starty + cropy]
    
    return acs


def inv_crop(target,center_tensor):
    padded_tensor = torch.zeros_like(target)
    pad_top = (padded_tensor.shape[0] - center_tensor.shape[0]) // 2
    pad_bottom = padded_tensor.shape[0] - center_tensor.shape[0] - pad_top
    pad_left = (padded_tensor.shape[1] - center_tensor.shape[1]) // 2
    pad_right = padded_tensor.shape[1] - center_tensor.shape[1] - pad_left
    pad_front = (padded_tensor.shape[2] - center_tensor.shape[2]) // 2
    pad_back = padded_tensor.shape[2] - center_tensor.shape[2] - pad_front
    pad_leftmost = (padded_tensor.shape[3] - center_tensor.shape[3]) // 2
    pad_rightmost = padded_tensor.shape[3] - center_tensor.shape[3] - pad_leftmost

    # 使用 pad 函数进行填充
    padded_tensor = F.pad(center_tensor, (pad_leftmost, pad_rightmost, pad_front, pad_back, pad_left, pad_right, pad_top, pad_bottom))
    return padded_tensor

def inv_crop_numpy(target, tensor):
    target_size = target.shape
    tensor_shape = np.array(tensor.shape)
    target_size = np.array(target_size)
    pad_sizes = np.maximum(target_size - tensor_shape, 0)
    pad_left = pad_sizes // 2
    pad_right = pad_sizes - pad_left
    padding = [(pad_left[i], pad_right[i]) for i in range(len(tensor_shape))]
    padded_tensor = np.pad(tensor, padding, mode='constant')
    return padded_tensor

def torch_crop(img, cropx, cropy):
    nb, c, x, y = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    if y>cropy and x>cropx:
        img = crop(img, cropx, cropy)
    elif y>cropy and x<cropx:
        img = crop(img, x, cropy)
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    elif y<cropy and x>cropx:
        img = crop(img, cropx, y)
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    else:
        target = torch.zeros(nb,c,cropx,cropy)
        img = inv_crop(target,img)
    return img

def pad_or_crop_tensor(input_tensor, target_shape):
    input_shape = input_tensor.shape
    pad_width = []

    # 计算每个维度需要填充的宽度或裁剪的宽度
    for i in range(len(target_shape)):
        diff = target_shape[i] - input_shape[i]

        # 计算前后需要填充的宽度或裁剪的宽度
        pad_before = max(0, diff // 2)
        pad_after = max(0, diff - pad_before)

        pad_width.append((pad_before, pad_after))

    # 使用numpy的pad函数进行填充或裁剪
    padded_tensor = np.pad(input_tensor, pad_width, mode='constant')

    # 裁剪张量为目标形状
    cropped_tensor = padded_tensor[:target_shape[0], :target_shape[1], :target_shape[2], :target_shape[3]]

    return cropped_tensor


def normalize(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= torch.min(img)
    img /= torch.max(img)
    return img


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


def normalize_complex(img):
    """ normalizes the magnitude of complex-valued image to range [0, 1] """
    abs_img = normalize(torch.abs(img))
    ang_img = normalize(torch.angle(img))
    return abs_img * torch.exp(1j * ang_img)


def normalize_l2(img):
    minv = np.std(img)
    img = img / minv
    return img


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def get_mask(config, caller):
    if caller == 'sde':
        if config.training.mask_type == 'low_frequency':
            mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acs + '.mat'
        elif config.training.mask_type == 'center':
            mask_file = 'mask/' +  config.training.mask_type + "_length" + config.training.acs + '.mat'
        else:
            mask_file = 'mask/' +  config.training.mask_type + "_acc" + config.training.acc \
                                                + '_acs' + config.training.acs + '.mat'
    elif caller == 'sample':
        mask_file = 'mask/' +  config.sampling.mask_type + "_acc" + config.sampling.acc \
                                                + '_acs' + config.sampling.acs + '.mat'
    elif caller == 'acs':
        mask_file = 'mask/low_frequency_acs18.mat'
    mask = scio.loadmat(mask_file)['mask']
    mask = mask.astype(np.complex)
    mask = np.expand_dims(mask, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).to(config.device)

    return mask
# def get_mask(config, caller):
#     if caller == 'sde':
#         if config.training.mask_type == 'low_frequency':
#             mask_file = 'mask/' +  config.training.mask_type + "_acs" + config.training.acc + '.mat'
#         else:
#             mask_file = 'mask_acs20/' +  config.training.mask_type + "_acc" + config.training.acc + '.mat'
#     elif caller == 'sample':
#         mask_file = 'mask_acs18/' +  config.sampling.mask_type + "_acc" + config.sampling.acc + '.mat'
#     mask = scio.loadmat(mask_file)['mask']
#     mask = mask.astype(np.complex128)
#     mask = np.expand_dims(mask, axis=0)
#     mask = np.expand_dims(mask, axis=0)
#     mask = torch.from_numpy(mask).to(config.device)

#     return mask


def ifftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fftshift(x, axes=None):
    assert torch.is_tensor(x) == True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def fft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def fft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny]).to(device)
    nx = torch.Tensor([nx]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(ny))
    return x


def FFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2)/np.sqrt(nx)
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)/np.sqrt(ny)
    return x

def FFT2c_3d(x):
    nb, nc, nt, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=3)
    x = np.transpose(x, [0, 1, 2, 4, 3])
    x = np.fft.fft(x, axis=-1)
    x = np.transpose(x, [0, 1, 2, 4, 3])
    x = np.fft.fftshift(x, axes=3)/np.sqrt(nx)
    x = np.fft.ifftshift(x, axes=4)
    x = np.fft.fft(x, axis=-1)
    x = np.fft.fftshift(x, axes=4)/np.sqrt(ny)
    return x


def ifft2c(x):
    device = x.device
    nb, nc, nt, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))
    return x


def ifft2c_2d(x):
    device = x.device
    nb, nc, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 3)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 3)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nx))
    x = ifftshift(x, axes=3)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(ny))
    return x

AXES_2d = (-2, -1)  # 对形如 [..., nx, ny] 的张量，在 x,y 维上做2D FFT

def fftn_2c(x: torch.Tensor) -> torch.Tensor:
    """Centered + orthonormal FFT over AXES (默认对最后两维 x,y)"""
    x = torch.fft.ifftshift(x, dim=AXES_2d)
    X = torch.fft.fftn(x, dim=AXES_2d, norm='ortho')
    X = torch.fft.fftshift(X, dim=AXES_2d)
    return X

def ifftn_2c(X: torch.Tensor) -> torch.Tensor:
    """Centered + orthonormal IFFT over AXES (默认对最后两维 x,y)"""
    X = torch.fft.ifftshift(X, dim=AXES_2d)
    x = torch.fft.ifftn(X, dim=AXES_2d, norm='ortho')
    x = torch.fft.fftshift(x, dim=AXES_2d)
    return x


def ifft2c_3d(x):
    device = x.device
    nb, nc, nz, nx, ny = x.size()
    ny = torch.Tensor([ny])
    ny = ny.to(device)
    nx = torch.Tensor([nx])
    nx = nx.to(device)
    nz = torch.Tensor([nz])
    nz = nz.to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 2, 4)
    x = torch.mul(fftshift(x, axes=2), torch.sqrt(nz))
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.ifft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.mul(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.ifft(x)
    x = torch.mul(fftshift(x, axes=4), torch.sqrt(ny))
    return x

def fft2c_3d(x):
    device = x.device
    nb, nc, nz, nx, ny = x.size()
    nx = torch.Tensor([nx]).to(device)
    ny = torch.Tensor([ny]).to(device)
    nz = torch.Tensor([nz]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 4)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nz))
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x

def fft2c_3d(x):
    device = x.device
    nb, nc, nz, nx, ny = x.size()
    nx = torch.Tensor([nx]).to(device)
    ny = torch.Tensor([ny]).to(device)
    nz = torch.Tensor([nz]).to(device)
    x = ifftshift(x, axes=2)
    x = torch.transpose(x, 2, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 2, 4)
    x = torch.div(fftshift(x, axes=2), torch.sqrt(nz))
    x = ifftshift(x, axes=3)
    x = torch.transpose(x, 3, 4)
    x = FFT.fft(x)
    x = torch.transpose(x, 3, 4)
    x = torch.div(fftshift(x, axes=3), torch.sqrt(nx))
    x = ifftshift(x, axes=4)
    x = FFT.fft(x)
    x = torch.div(fftshift(x, axes=4), torch.sqrt(ny))
    return x

AXES = (-3, -2, -1)

def fftn_c(x: torch.Tensor):
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=AXES), dim=AXES, norm='ortho'),
        dim=AXES,
    )

def ifftn_c(X: torch.Tensor):
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(X, dim=AXES), dim=AXES, norm='ortho'),
        dim=AXES,
    )


def fftc_1d(x: torch.Tensor, dim: int, backend: str = "torch", enforce_complex: bool = True):
    """
    Centered 1D FFT along `dim` with /sqrt(N) normalization.
    e.g: kspace_line = fftc_1d(image_data, dim=2)
    """
    if enforce_complex and not torch.is_complex(x):
        x = x.to(torch.complex64)

    device = x.device
    dim = dim if dim >= 0 else x.ndim + dim
    n = torch.tensor(x.size(dim), device=device, dtype=x.real.dtype)

    x = ifftshift(x, axes=dim)

    if backend == "torch":
        x = torch.fft.fft(x, dim=dim, norm=None)
    elif backend == "project":
        x = x.movedim(dim, -1)
        x = FFT.fft(x)
        x = x.movedim(-1, dim)
    else:
        raise ValueError("backend must be 'torch' or 'project'.")

    x = fftshift(x, axes=dim)
    return x / n.sqrt()

def ifftc_1d(X: torch.Tensor, dim: int, backend: str = "torch", enforce_complex: bool = True):
    """
    Centered 1D IFFT along `dim` with *sqrt(N) normalization.
    image_recon = ifftc_1d(kspace_line, dim=-1) 最后维度执行ifftc
    """
    if enforce_complex and not torch.is_complex(X):
        X = X.to(torch.complex64)

    device = X.device
    dim = dim if dim >= 0 else X.ndim + dim
    n = torch.tensor(X.size(dim), device=device, dtype=X.real.dtype)

    X = ifftshift(X, axes=dim)

    if backend == "torch":
        X = torch.fft.ifft(X, dim=dim, norm=None)
    elif backend == "project":
        X = X.movedim(dim, -1)
        X = FFT.ifft(X)
        X = X.movedim(-1, dim)
    else:
        raise ValueError("backend must be 'torch' or 'project'.")

    X = fftshift(X, axes=dim)
    return X * n.sqrt()

def IFFT2c(x):
    nb, nc, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=2)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 3, 2])
    x = np.fft.fftshift(x, axes=2) * np.sqrt(nx)  
    x = np.fft.ifftshift(x, axes=3)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=3)*np.sqrt(ny)
    return x

def IFFT2c_3d(x):
    nb, nc, nt, nx, ny = np.shape(x)
    x = np.fft.ifftshift(x, axes=3)
    x = np.transpose(x, [0, 1, 2, 4, 3])
    x = np.fft.ifft(x, axis=-1)
    x = np.transpose(x, [0, 1, 2, 4, 3])
    x = np.fft.fftshift(x, axes=3) * np.sqrt(nx)  
    x = np.fft.ifftshift(x, axes=4)
    x = np.fft.ifft(x, axis=-1)
    x = np.fft.fftshift(x, axes=4)*np.sqrt(ny)
    return x

def Emat_xyt(b, inv, csm, mask):
    if csm == None:
        if inv:
            b = r2c(b) * mask
            if b.ndim == 4:
                b = ifft2c_2d(b)
            else:
                b = ifft2c_3d(b)
            x = c2r(b)
        else:
            b = r2c(b)
            if b.ndim == 4:
                b = fft2c_2d(b) * mask
            else:
                b = fft2c_3d(b) * mask
            x = c2r(b)
    else:
        if inv:
            csm = r2c(csm)
            x = r2c(b) * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c_3d(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)
            x = c2r(x)

        else:
            csm = r2c(csm)
            b = r2c(b)
            b = b*csm
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c_3d(b)
            x = mask*b
            x = c2r(x)
            
    return x




def SS_H(z,csm):
    z = r2c(z)
    csm = r2c(csm)
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)
    z = z*csm
    return c2r(z)

def S_H(z,csm):
    z = r2c(z)
    csm = r2c(csm)
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)
    return c2r(z)

def SS_H_hat(z,csm):
    z = r2c(z)
    z = ifft2c_2d(z)
    csm = r2c(csm)
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)
    z = z*csm
    z = fft2c_2d(z)
    return c2r(z)

def S_H_hat(z,csm):
    z = r2c(z)
    z = ifft2c_2d(z)
    csm = r2c(csm)
    z = torch.sum(z*torch.conj(csm),dim=1,keepdim=True)
    z = fft2c_2d(z)
    return c2r(z)

def ch_to_nb(z,filt=None):
    z = r2c(z)
    if filt==None:
        z = torch.permute(z,(1,0,2,3))
    else:
        z = torch.permute(z,(1,0,2,3))/filt
    return c2r(z)

def Emat_xyt_complex(b, inv, csm, mask):
    if csm == None: 
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c(x)
            x = x*torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)

        else:
            b = b*csm # 
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c(b)
            x = mask*b
    return x

def Emat_zxy_complex(b, inv, csm, mask):
    if csm == None: 
        if inv:
            b = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(b)
            else:
                x = ifft2c_3d(b)
        else:
            if b.ndim == 4:
                x = fft2c_2d(b) * mask
            else:
                x = fft2c_3d(b) * mask
    else:
        if inv:
            x = b * mask
            if b.ndim == 4:
                x = ifft2c_2d(x)
            else:
                x = ifft2c_3d(x) # 1x12x171x128x256

            x = x * torch.conj(csm)
            x = torch.sum(x, 1)
            x = torch.unsqueeze(x, 1)

        else:
            b = b*csm # 
            if b.ndim == 4:
                b = fft2c_2d(b)
            else:
                b = fft2c_3d(b)
            x = mask*b
    return x

def r2c(x):
    re, im = torch.chunk(x, 2, 1)
    x = torch.complex(re, im)
    return x


def c2r(x):
    x = torch.cat([torch.real(x), torch.imag(x)], 1)
    return x


def sos(x):
    xr, xi = torch.chunk(x, 2, 1)
    x = torch.pow(torch.abs(xr), 2)+torch.pow(torch.abs(xi), 2)
    x = torch.sum(x, dim=1)
    x = torch.pow(x, 0.5)
    x = torch.unsqueeze(x, 1)
    return x


def Abs(x):
    x = r2c(x)
    return torch.abs(x)


def l2mean(x):
    result = torch.mean(torch.pow(torch.abs(x), 2))

    return result


def TV(x, norm='L1'):
    nb, nc, nx, ny = x.size()
    Dx = torch.cat([x[:, :, 1:nx, :], x[:, :, 0:1, :]], 2)
    Dy = torch.cat([x[:, :, :, 1:ny], x[:, :, :, 0:1]], 3)
    Dx = Dx - x
    Dy = Dy - x
    tv = 0
    if norm == 'L1':
        tv = torch.mean(torch.abs(Dx)) + torch.mean(torch.abs(Dy))
    elif norm == 'L2':
        Dx = Dx * Dx
        Dy = Dy * Dy
        tv = torch.mean(Dx) + torch.mean(Dy)
    return tv

def stdnormalize(x):
    x = r2c(x)
    result = c2r(x)/torch.std(x)
    return result

def to_null_space(x,mask,csm):
    Aobj = Aclass(csm, mask, torch.tensor(.01).cuda())
    Rhs = Emat_xyt(x, False, csm, mask)
    Rhs = Emat_xyt(Rhs, True, csm, mask)

    x_null = x - myCG(Aobj, Rhs, x, 5) 

    #x = ifft2c_2d(fft2c_2d(r2c(x))*(1-mask))
    #chc = torch.conj(r2c(csm))*r2c(csm)+1e-5
    #x = torch.conj(r2c(csm))/chc*x
    #x = c2r(torch.sum(x, keepdim=True, dim=1))
    #x = stdnormalize(x)
    #x = torch.min(torch.abs(chc))*c2r(x)
    return x_null 


                


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    A^{T}A * X + \lamda *X
    """
    def __init__(self, csm, mask, lam):
        self.pixels = mask.shape[0] * mask.shape[1]
        self.mask = mask
        self.csm = csm
        self.SF = torch.complex(torch.sqrt(torch.tensor(self.pixels).float()), torch.tensor(0.).float())
        self.lam = lam

    def myAtA(self, img):
        
        x = Emat_xyt(img, False, self.csm, self.mask)
        x = Emat_xyt(x, True, self.csm, self.mask)
        
        return x + self.lam * img


def myCG(A, Rhs, x0, it):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    #print('Rhs1', Rhs.shape, Rhs.dtype) #Rhs1.shape torch.Size([2, 256, 232])

    x0 = torch.zeros_like(Rhs)
    Rhs = r2c(Rhs) + A.lam * r2c(x0)
    
    #x = torch.zeros_like(Rhs)
    x = r2c(x0)
    i = 0
    r = Rhs - r2c(A.myAtA(x0))
    p = r
    rTr = torch.sum(torch.conj(r)*r).float()

    while i < it:
        Ap = r2c(A.myAtA(c2r(p)))
        alpha = rTr / torch.sum(torch.conj(p)*Ap).float()
        alpha = torch.complex(alpha, torch.tensor(0.).float().cuda())
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = torch.sum(torch.conj(r)*r).float()
        beta = rTrNew / rTr
        beta = torch.complex(beta, torch.tensor(0.).float().cuda())
        p = r + beta * p
        i = i + 1
        rTr = rTrNew

    return c2r(x)


def restore_checkpoint(ckpt_dir, state, device):
    # if not tf.io.gfile.exists(ckpt_dir):
    #     tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    #     logging.warning(f"No checkpoint found at {ckpt_dir}. "
    #                     f"Returned the same state as input")
    #     return state
    # else:

    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def complex_kernel_forward(filter, i):
    filter = torch.squeeze(filter[i])
    filter_real = torch.real(filter)
    filter_img = torch.imag(filter)
    kernel_real = torch.cat([filter_real, -filter_img], 1)
    kernel_imag = torch.cat([filter_img, filter_real], 1)
    kernel_complex = torch.cat([kernel_real, kernel_imag], 0)
    return kernel_complex


def conv2(x1, x2):
    return F.conv2d(x1, x2, padding='same')


def ksp2float(ksp, i):
    kdata = torch.squeeze(ksp[i])  # nb,nc,nx,ny
    if len(kdata.shape) == 3:
        kdata = torch.unsqueeze(kdata, 0)

    kdata_float = torch.cat([torch.real(kdata), torch.imag(kdata)], 1)
    return kdata_float


def spirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc, nx, ny
    """
    nb = ksp.shape[0]

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(kernel, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))

    re, im = torch.chunk(res_i, 2, 1)
    res = torch.complex(re, im) - ksp
    return res


def adjspirit(kernel, ksp):
    """
    :param kernel: nb, nc, nc_s, kx, ky
    :param ksp: nb, nc, nx, ny
    :return: SPIRiT output: nb, nc_s, nx, ny
    """

    nb = kernel.shape[0]
    filter = torch.permute(kernel, (0, 2, 1, 3, 4))
    filter = torch.conj(filter.flip(dims=[-2, -1]))

    if len(ksp.shape) == 5:
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))
        res_i = torch.stack([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)
    else:
        res_i = torch.cat([conv2(ksp2float(ksp, i), complex_kernel_forward(filter, i)) for i in range(nb)], 0)

    if len(ksp.shape) == 5:
        res_i = torch.permute(res_i, (0, 2, 1, 3, 4))
        ksp = torch.permute(ksp, (0, 2, 1, 3, 4))

    re, im = torch.chunk(res_i, 2, 1)

    res = torch.complex(re, im) - ksp

    return res


def dot_batch(x1, x2):
    batch = x1.shape[0]
    res = torch.reshape(x1 * x2, (batch, -1))
    # res = torch.reshape(x1 * x2, (-1, 1))
    return torch.sum(res, 1)


class ConjGrad:
    def __init__(self, A, rhs, max_iter=5, eps=1e-10):
        self.A = A
        self.b = rhs
        self.max_iter = max_iter
        self.eps = eps

    def forward(self, x):
        x = CG(x, self.b, self.A, max_iter=self.max_iter, eps=self.eps)
        return x

def CG(x, b, A, max_iter, eps):
    b = b + eps*x
    r = b - A.A(x)
    p = r
    rTr = dot_batch(torch.conj(r), r)
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    num_iter = 0
    for iter in range(max_iter):
        if rTr.abs().max() < eps:
            break
        Ap = A.A(p)
        alpha = rTr / dot_batch(torch.conj(p), Ap)
        alpha = torch.reshape(alpha, reshape)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = dot_batch(torch.conj(r), r)
        beta = rTrNew / rTr
        beta = torch.reshape(beta, reshape)
        p = r + beta * p
        rTr = rTrNew

        num_iter += 1
    return x

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')
    return np.dot(A.T.conj(), A)


def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            # res[:, count, :] = np.reshape(
            #     im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))
            count += 1
    return res

def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))
    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1

    idxY = np.where(dummyK)
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty) # fast 1s

    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)


def spirit_calibrate(acs, kSize, lamda=0.001, filtering=False, verbose=True): # lamda=0.01
    nCoil = acs.shape[-1]
    AtA = dat2AtA(acs,kSize)
    if filtering: # singular value threshing
        if verbose:
            ic('prefiltering w/ opth')
        U,s,Vh = svd(AtA, full_matrices=False)
        k = optht(AtA, sv=s, sigma=None)
        if verbose:
            print('{}/{} kernels used'.format(k, len(s)))
        AtA= (U[:, :k] * s[:k] ).dot( Vh[:k,:])
        
    spirit_kernel = np.zeros((nCoil,nCoil,*kSize),dtype='complex128')
    for c in tqdm(range(nCoil)):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)
        spirit_kernel[c] = np.transpose(tmp,[2,0,1])
    spirit_kernel = np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!
    GOP = np.transpose(spirit_kernel[::-1,::-1],[3,2,0,1])
    GOP = GOP.copy()
    for n in range(nCoil):
        GOP[n,n,kSize[0]//2,kSize[1]//2] = -1  
    return spirit_kernel


class Aclass_spirit:
    def __init__(self, kernel, mask, lam):
        self.kernel = kernel
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        ksp = spirit(self.kernel, ksp)
        ksp = adjspirit(self.kernel, ksp)
        return ksp

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * self.mask * ksp
        return res


class Aclass_spirit_proj:
    def __init__(self, kernel, mask, lam1, lam2, xi):
        self.kernel = kernel
        self.mask = mask
        self.lam1 = lam1
        self.lam2 = lam2
        self.xi = xi

    def ATA(self, ksp):
        ksp = spirit(self.kernel, ksp)
        ksp = adjspirit(self.kernel, ksp)
        return ksp

    def A(self, ksp):
        res = self.mask*ksp + self.ATA(ksp) * self.lam1 + self.lam2 * torch.sum(self.xi * ksp)*self.xi
        return res
    

def sense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c_2d(m * csm)
    return res - ksp

def sense3d(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_3d(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c_3d(m * csm)
    return res - ksp

def adjsense(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_2d(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c_2d(m * csm)
    return res - ksp

def adjsense3d(csm, ksp):
    """
    :param csm: nb, nc, nx, ny
    :param ksp: nb, nc, nt, nx, ny
    :return: SENSE output: nb, nt, nx, ny
    """
    m = torch.sum(ifft2c_3d(ksp) * torch.conj(csm), 1, keepdim=True)
    res = fft2c_3d(m * csm)
    return res - ksp

class Aclass_sense:
    def __init__(self, csm, mask, lam):
        self.s = csm
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        Ax = sense(self.s, ksp)
        AHAx = adjsense(self.s, Ax)
        return AHAx

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res

class Aclass_sense3d:
    def __init__(self, csm, mask, lam):
        self.s = csm
        self.mask = 1 - mask
        self.lam = lam

    def ATA(self, ksp):
        Ax = sense3d(self.s, ksp)
        AHAx = adjsense3d(self.s, Ax)
        return AHAx

    def A(self, ksp):
        res = self.ATA(ksp * self.mask) * self.mask + self.lam * ksp
        return res

class Aclass_sensev2:
    def __init__(self, csm, mask, lam):
        self.s = csm
        self.mask = mask
        self.lam = lam

    def ATA(self, x):
        Ax = Emat_xyt(x, False, self.s, self.mask)
        AHAx = Emat_xyt(Ax, True, self.s, self.mask)
        return AHAx

    def A(self, x):
        res = self.ATA(x) + self.lam * x
        return res



class Aclass_spiritv2:
    def __init__(self, kernel, mask, lam1, lam2):
        self.kernel = kernel
        self.mask = mask
        self.lam1 = lam1
        self.lam2 = lam2

    def ATA(self, ksp):
        ksp = spirit(self.kernel, ksp)
        ksp = adjspirit(self.kernel, ksp)
        return ksp

    def A(self, ksp):
        res = self.lam1*self.ATA(ksp) + self.mask*ksp + self.lam2 * ksp
        return res

def cgSPIRiTv2(x0, ksp, kernel, mask, niter, lam1, lam2):
    Aobj = Aclass_spiritv2(kernel, mask, lam1, lam2)
    y = ksp
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    return x

class Aclass_Self:
    def __init__(self, kernel, lam):
        self.kernel = kernel
        self.lam = lam

    def ATA(self, ksp):
        ksp = spirit(self.kernel, ksp)
        ksp = adjspirit(self.kernel, ksp)
        return ksp

    def A(self, ksp):
        res = self.ATA(ksp) + self.lam * ksp
        return res

def cgSELF(x0, kernel, niter, lam):
    Aobj = Aclass_Self(kernel, lam)
    y = 0
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    return x

def cgSENSEv2(x0, ksp, csm, mask, niter, lam): 
    x0 = c2r(x0)
    ksp = c2r(ksp)
    csm = c2r(csm)
    Aobj = Aclass_sensev2(csm, mask, lam)
    y = Emat_xyt(ksp, True, csm, 1)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    return r2c(x)

def cgSENSE(x0, ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c_2d(x) * torch.conj(csm), 1, keepdim=True)
    return x, res

def cgSENSE3D(x0, ksp, csm, mask, niter, lam):
    Aobj = Aclass_sense3d(csm, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    x = x * (1 - mask) + ksp
    res = torch.sum(ifft2c_3d(x) * torch.conj(csm), 1, keepdim=True)
    return x, res
    
def cgSPIRiT(x0, ksp, kernel, mask, niter, lam):
    Aobj = Aclass_spirit(kernel, mask, lam)
    y = - (1 - mask) * Aobj.ATA(ksp) + lam * (1 - mask)* x0
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    x =  x * (1 - mask) + ksp
    return x

def cgSPIRiT_Proj(x0, xi, ksp, kernel, mask, niter, lam1, lam2):
    Aobj = Aclass_spirit_proj(kernel, mask, lam1, lam2, xi)
    y =  mask * ksp + lam2 * torch.sum(xi * x0)
    cg_iter = ConjGrad(Aobj, y, max_iter=niter)
    x = cg_iter.forward(x=x0)
    # x =  x * (1 - mask) + ksp
    return x

def SPIRiT_Aobj(kernel,ksp):

    ksp = spirit(kernel, ksp)
    ksp = adjspirit(kernel, ksp)
    return ksp


def add_noise(x, snr):
    x_ = x.view(x.shape[0], -1)
    x_power = torch.sum(torch.pow(torch.abs(x_), 2), dim=1, keepdim=True) / x_.shape[1]
    snr = 10 ** (snr / 10)
    noise_power = x_power / snr
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    noise_power = torch.reshape(noise_power, reshape)
    if x.dtype == torch.float32:
        noise = torch.sqrt(noise_power) * torch.randn(x.size(), device=x.device)
    else:
        noise = torch.sqrt(0.5 * noise_power) * (torch.complex(torch.randn(x.size(), device=x.device),
                                                               torch.randn(x.size(), device=x.device)))
    return x + noise


def blur_and_noise(x, kernel_size=7, sig=0.1, snr=10):
    x_org = x
    transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sig)
    if x.dtype == torch.float32:
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))
    else:
        x = c2r(x)
        x_ = torch.reshape(x, (-1, x.shape[-2], x.shape[-1]))

    x_blur = transform(x_)
    x_blur = torch.reshape(x_blur, x.shape)
    x_blur_noise = add_noise(x_blur, snr=snr)
    if x_org.dtype == torch.float32:
        return x_blur_noise
    else:
        return r2c(x_blur_noise)

def matmul_cplx(x1, x2):
    return torch.view_as_complex(
        torch.stack((x1.real @ x2.real - x1.imag @ x2.imag, x1.real @ x2.imag + x1.imag @ x2.real), dim=-1))

def Gaussian_mask(nx, ny, Rmax, t, Fourier=True):
    if nx % 2 == 0:
        ix = np.arange(-nx//2, nx//2)
    else:
        ix = np.arange(-nx//2, nx//2 + 1)

    if ny % 2 == 0:
        iy = np.arange(-ny//2, ny//2)
    else:
        iy = np.arange(-ny//2, ny//2 + 1)

    wx = Rmax * ix / (nx / 2)
    wy = Rmax * iy / (ny / 2)

    rwx, rwy = np.meshgrid(wx, wy)
    # R = np.exp(-(rwx ** 2 + rwy ** 2) / (2 * t ** 2)) / (np.sqrt(2 * np.pi) * t)
    if Fourier:
        R = np.exp(-((rwx ** 2 + rwy ** 2)* t ** 2) / 2 )
    else:
        R = np.exp(-(rwx ** 2 + rwy ** 2) / (2 * t ** 2))
        
    W = R.astype(np.float32)
    # W = W/np.max(W)

    return W


def ESPIRiT_calib(ksp, i, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp[i]) 
    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm

def ESPIRiT_calib_parallel(ksp, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp) 
    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm.unsqueeze(0)

def ESPIRiT_calib_prescan(ksp_prescan, ksp, i, gpu_id, calib=24, crop=0):
    kdata = torch.squeeze(ksp_prescan[i]) 
    calib = kdata.shape[-1]
    zpad = T.CenterCrop((int(ksp.shape[-2]),int(ksp.shape[-1])))
    kdata = zpad(kdata)

    ksp_gpu = cu_from_dlpack(to_dlpack(kdata))
    csm = MR.EspiritCalib(ksp_gpu, calib_width=calib, crop=crop, device=sp.Device(gpu_id), show_pbar=False).run()
    csm = from_dlpack(csm.toDlpack())
    return csm

def matlab_style_reshape(tensor, new_shape):
    permuted_tensor = tensor.permute(*reversed(range(tensor.dim())))
    reshaped_tensor = permuted_tensor.reshape(*reversed(new_shape))
    final_tensor = reshaped_tensor.permute(*reversed(range(len(new_shape))))

    return final_tensor

def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def patch2hank_complex_single(inp, M, N, NC, m, n):
    inp = inp.permute(1, 2, 0)
    #print('inp', inp.shape) 
    out = torch.zeros(((M-m+1)*(N-n+1), m*n, NC), dtype=torch.complex64, device=inp.device)
    #print('out', out.shape)
    inc = 0
    for niter in range(n):
        for miter in range(m):
            inc += 1
            out[:, inc-1, :] = matlab_style_reshape(inp[miter:M-m+miter+1, niter:N-n+niter+1, :], [(M-m+1)*(N-n+1), NC])
    out = matlab_style_reshape(out, [out.shape[0], -1])
    return out

def hank2patch_complex_single(inp, M, N, NC, m, n):
    out = torch.zeros((M, N, NC), dtype=torch.complex64, device=inp.device)
    W = torch.zeros((M, N, NC), dtype=torch.complex64, device=inp.device)
    inp = matlab_style_reshape(inp, [inp.shape[0], -1, NC])
    inc = 0
    for niter in range(n):
        for miter in range(m):
            inc += 1
            out[miter:M-m+miter+1, niter:N-n+niter+1, :] += matlab_style_reshape(inp[:, inc-1, :], [(M-m+1), (N-n+1), NC])
            W[miter:M-m+miter+1, niter:N-n+niter+1, :] += 1
    out = out / W
    #print('out2', out.shape)
    return out.permute(2, 0, 1)

def match_matlab_svd(A):
    U, S, V = torch.svd(A, some=False) 
    max_abs_cols = torch.argmax(torch.abs(V), dim=0)
    signs = torch.sign(V[max_abs_cols, torch.arange(V.shape[1])])
    U *= signs
    V *= signs.unsqueeze(0)
    return U, S, V
 
    
def MySVD(A):
    print('A', A.shape)
    m, n = A.shape
#    if 2 * m < n:
#        AAT = torch.matmul(A, A.t())
#        S, Sigma2, D = match_matlab_svd(AAT)
#        print('Sigma1', S.shape, Sigma2.shape, D.shape)
#        #Sigma2 = torch.diag(Sigma2)
#        #print('Sigma2', Sigma2.shape)
#        #V = torch.sqrt(Sigma2)
#        V = Sigma2
#        #print('V', V.shape)
#        tol = max(A.shape) * torch.finfo(S.dtype).eps * torch.max(V)
#        R = torch.sum(V > tol)
#        V = V[:R]
#        S = S[:, :R]
#        D = torch.matmul(torch.matmul(A.t(), S), (1+0j / V))
#        
#        print('AS', A.t().shape, S.shape, V.shape, D.shape, V)
#        #V = torch.diag(V)
#        #return S, V, D, Sigma2

    if m > 2 * n:        
        B = A.mH
        AAT = torch.matmul(B, B.mH)
        S, Sigma2, D = torch.svd(AAT)#torch.linalg.svd(AAT, full_matrices=True)

        V = torch.sqrt(Sigma2)
        tol = max(B.shape) * torch.finfo(V.dtype).eps * torch.max(V)
        R = torch.sum(V > tol)

        V = V[:R]#.clone()      
        S = S[:, :R]
        
        D = torch.matmul(torch.matmul(B.mH, S), torch.diag(torch.div(torch.tensor(1).to(V.dtype), V)).to(torch.complex64))

        mid = D.clone()
        D = S
        S = mid
    return S, V, D, Sigma2


def orthogonal_component(u, v):
    """
    计算高维张量 u 在 v 垂直方向上的分量
    :param u: 待分解的张量 (Tensor)，形状为 (N, C, H, W)
    :param v: 参考方向张量 (Tensor)，形状为 (N, C, H, W)
    :return: u 在 v 垂直方向上的分量 (Tensor)
    """
    # 点积按元素相乘后求和，保持批处理支持
    dot_product_uv = torch.sum(u * v, dim=(1, 2, 3), keepdim=True)  # u · v
    norm_squared_v = torch.sum(v * v, dim=(1, 2, 3), keepdim=True)  # ||v||^2

    # 检查零向量，避免除零错误
    if torch.any(norm_squared_v == 0):
        raise ValueError("v contains zero vectors, cannot compute orthogonal components.")
    
    # 计算投影分量
    projection = (dot_product_uv / norm_squared_v) * v

    # 计算垂直分量
    orthogonal = u - projection
    return orthogonal


def decompose_vector(v: torch.Tensor, d: torch.Tensor):
    """将向量 v 沿 d 方向分解为垂直和切向分量"""
    d_norm = d / torch.norm(d, dim=-1, keepdim=True)  # 归一化方向向量
    v_parallel = torch.einsum('...i,...i->...', v, d_norm)[..., None] * d_norm  # 切向分量
    v_perpendicular = v - v_parallel  # 垂直分量
    return v_parallel, v_perpendicular


class EspiritCalib(sp.app.App):
    """ESPIRiT calibration.

    Currently only supports outputting one set of maps.

    Args:
        ksp (array): k-space array of shape [num_coils, n_ndim, ..., n_1]
        calib (tuple of ints): length-2 image shape.
        thresh (float): threshold for the calibration matrix.
        kernel_width (int): kernel width for the calibration matrix.
        max_power_iter (int): maximum number of power iterations.
        device (Device): computing device.
        crop (int): cropping threshold.

    Returns:
        array: ESPIRiT maps of the same shape as ksp.

    References:
        Martin Uecker, Peng Lai, Mark J. Murphy, Patrick Virtue, Michael Elad,
        John M. Pauly, Shreyas S. Vasanawala, and Michael Lustig
        ESPIRIT - An Eigenvalue Approach to Autocalibrating Parallel MRI:
        Where SENSE meets GRAPPA.
        Magnetic Resonance in Medicine, 71:990-1001 (2014)

    """
    def __init__(self, ksp, calib_width=24,
                 thresh=0.02, kernel_width=6, crop=0,
                 max_iter=90, device=sp.cpu_device,
                 output_eigenvalue=False, show_pbar=True):
        self.device = sp.Device(device)
        self.output_eigenvalue = output_eigenvalue
        self.crop = crop

        img_ndim = ksp.ndim - 1
        num_coils = len(ksp)
        with sp.get_device(ksp):
            # Get calibration region
            calib_shape = [num_coils] + [calib_width] * img_ndim
            calib = sp.resize(ksp, calib_shape)
            calib = sp.to_device(calib, device)

        xp = self.device.xp
        with self.device:
            # Get calibration matrix.
            # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
            mat = sp.array_to_blocks(
                calib, [kernel_width] * img_ndim, [1] * img_ndim)
            mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
            mat = mat.transpose([1, 0, 2])
            mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])

            # Perform SVD on calibration matrix
            _, S, VH = xp.linalg.svd(mat, full_matrices=False)
            VH = VH[S > thresh * S.max(), :]

            # Get kernels
            num_kernels = len(VH)
            kernels = VH.reshape(
                [num_kernels, num_coils] + [kernel_width] * img_ndim)
            img_shape = ksp.shape[1:]

            # Get covariance matrix in image domain
            AHA = xp.zeros(img_shape[::-1] + (num_coils, num_coils),
                           dtype=ksp.dtype)
            for kernel in kernels:
                img_kernel = sp.ifft(sp.resize(kernel, ksp.shape),
                                     axes=range(-img_ndim, 0))
                aH = xp.expand_dims(img_kernel.T, axis=-1)
                a = xp.conj(aH.swapaxes(-1, -2))
                AHA += aH @ a

            AHA *= (sp.prod(img_shape) / kernel_width**img_ndim)
            self.mps = xp.ones(ksp.shape[::-1] + (1, ), dtype=ksp.dtype)

            def forward(x):
                with sp.get_device(x):
                    return AHA @ x

            def normalize(x):
                with sp.get_device(x):
                    return xp.sum(xp.abs(x)**2,
                                  axis=-2, keepdims=True)**0.5

            alg = sp.alg.PowerMethod(
                forward, self.mps, norm_func=normalize,
                max_iter=max_iter)

        super().__init__(alg, show_pbar=show_pbar)

    def _output(self):
        xp = self.device.xp
        with self.device:
            # Normalize phase with respect to first channel
            mps = self.mps.T[0]
            mps *= xp.conj(mps[0] / xp.abs(mps[0]))

            # Crop maps by thresholding eigenvalue
            max_eig = self.alg.max_eig.T[0]
            mps *= max_eig > self.crop

        if self.output_eigenvalue:
            return mps, max_eig
        else:
            return mps
