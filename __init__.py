import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from torch.utils.data import Subset
import numpy as np
import sigpy.mri as mr
import sigpy as sp
import random
import mat73
import math
import pickle
import h5py
from torch.utils.data import Dataset, DataLoader
from utils import *
import dill

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class FastMRIKneeDataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIKneeDataSet, self).__init__()
        self.config = config
        if mode == 'train':
            self.kspace_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_34/T1_data_34/'
            self.maps_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_34/Output_maps_34/'
            # self.kspace_dir = '/data1/zhuoxu/data/fastMRI/multicoil_train/kspace/'
            # self.maps_dir = '/data1/zhuoxu/data/fastMRI/multicoil_train/maps/'            
        elif mode == 'test':
            self.kspace_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_test/T1_data/'
            self.maps_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_test/output_maps/'
        elif mode == 'sample':
            self.kspace_dir = '/data1/data1_congcong/data/fastMRI/test/fastMRI_knee_sample/T1_data/'
            self.maps_dir = '/data1/data1_congcong/data/fastMRI/test/fastMRI_knee_sample/output_maps/'
        elif mode == 'datashift':
            self.kspace_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/_brain/brain_T2/'
            self.maps_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_brain/output_maps/'
        else:
            raise NotImplementedError

        self.mode = mode
        self.crop_size = config.data.image_size
        self.file_list = get_all_files(self.kspace_dir)
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(
                self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:
                if self.mode != 'sample':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 6)
                else:
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0])

        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 被试者编号
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # 被试者扫描的帧数编号
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] +
                self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        maps_file = os.path.join(self.maps_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(maps_file, 'r') as data:
            # 去掉前6帧
            if self.mode != 'sample':
                slice_idx = slice_idx + 6
            maps_idx = data['s_maps'][slice_idx]
            maps_idx = np.expand_dims(maps_idx, 0)
            maps_idx = np_crop(maps_idx, cropc=maps_idx.shape[1], cropx=self.crop_size, cropy=self.crop_size)
            # maps_idx = np.squeeze(maps_idx, 0)
            maps = np.asarray(maps_idx)
            maps = torch.from_numpy(maps)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.kspace_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            ksp_idx = data['kspace'][slice_idx] # 15x640x368
            ksp_idx = np.expand_dims(ksp_idx, 0)
            ksp_idx = np_crop(IFFT2c(ksp_idx), cropc=ksp_idx.shape[1], cropx=self.crop_size, cropy=self.crop_size)
            ksp_idx = FFT2c(ksp_idx)
            # ksp_idx = np.squeeze(ksp_idx, 0)
            ksp_idx = torch.from_numpy(ksp_idx)

            # img_idx = Emat_xyt_complex(ksp_idx, True, maps, 1)
            # img_idx = normalize_complex(img_idx)
            # ksp_idx = Emat_xyt_complex(img_idx, False, maps, 1)

            

        ksp_idx = torch.squeeze(ksp_idx, 0)
        maps = torch.squeeze(maps, 0)
        
        return ksp_idx, maps

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))

class DataSet_3D(Dataset):
    def __init__(self, config, mode):
        super(DataSet_3D, self).__init__()
        self.config = config
        if mode == 'train':
            self.kspace_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_34/T1_data_34/'
            self.maps_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_knee_34/Output_maps_34/'
            # self.kspace_dir = '/data1/zhuoxu/data/fastMRI/multicoil_train/kspace/'
            # self.maps_dir = '/data1/zhuoxu/data/fastMRI/multicoil_train/maps/'            
        elif mode == 'test':
            self.kspace_dir = '/data0/congcong/data/knee_data_tmp/'

        elif mode == 'sample':
            self.kspace_dir = '/data0/congcong/data/3D_data/h5data/'

        elif mode == 'datashift':
            self.kspace_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/_brain/brain_T2/'
            self.maps_dir = '/data0/zhuoxu/data/fastMRI/T1_knee_1000/fastMRI_brain/output_maps/'
        else:
            raise NotImplementedError

        self.mode = mode
        self.crop_size = config.data.image_size
        self.file_list = get_all_files(self.kspace_dir)
        self.num_slices = np.zeros((len(self.file_list,)), dtype=int)
        for idx, file in enumerate(self.file_list):
            print('Input file:', os.path.join(
                self.kspace_dir, os.path.basename(file)))
            with h5py.File(os.path.join(self.kspace_dir, file), 'r') as data:
                if self.mode != 'sample':
                    self.num_slices[idx] = int(np.array(data['kspace']).shape[0] - 6)
                else:
                    data = np.array(data['kspace'])
                    data = data[None, ...] # 1x12x171x136x156
                    print("org shape:", data.shape)
                    self.num_slices[idx] = int(np.array(data).shape[0]) # 1*64*384*384*15


        # Create cumulative index for mapping
        self.slice_mapper = np.cumsum(self.num_slices) - 1  # Counts from '0'

    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 被试者编号
        scan_idx = int(np.where((self.slice_mapper - idx) >= 0)[0][0])
        # 被试者扫描的帧数编号
        slice_idx = int(idx) if scan_idx == 0 else \
            int(idx - self.slice_mapper[scan_idx] +
                self.num_slices[scan_idx] - 1)

        # Load maps for specific scan and slice
        # maps_file = os.path.join(self.maps_dir, os.path.basename(self.file_list[scan_idx]))
        # with h5py.File(maps_file, 'r') as data:
        #     # 去掉前6帧
        #     if self.mode != 'sample':
        #         slice_idx = slice_idx + 6
        #     maps_idx = data['s_maps'][slice_idx]
        #     maps_idx = np.expand_dims(maps_idx, 0)
        #     maps_idx = np_crop(maps_idx, cropc=maps_idx.shape[1], cropx=self.crop_size, cropy=self.crop_size)
        #     # maps_idx = np.squeeze(maps_idx, 0)
        #     maps = np.asarray(maps_idx)
        #     maps = torch.from_numpy(maps)

        # Load raw data for specific scan and slice
        raw_file = os.path.join(self.kspace_dir, os.path.basename(self.file_list[scan_idx]))
        with h5py.File(raw_file, 'r') as data:
            data = np.array(data['kspace'])
            data = data[None, ...] # 1x64x384x384x15
            print("getitem shape before crop:", data.shape) # 1x12x171x136x156
            ksp_idx = data[slice_idx] 
            ksp_idx = ksp_idx[None, ...]       # 1x12x171x136x156
            ksp_idx = torch.from_numpy(ksp_idx)
            # ksp_idx = fftc_1d(ksp_idx, dim=2) # 1x12x171x136x156
            # ksp_idx = np_crop(IFFT2c_3d(ksp_idx), cropc=ksp_idx.shape[2], cropx=self.crop_size, cropy=self.crop_size)
            # ksp_idx = FFT2c_3d(ksp_idx) # 1x15x64x384x384
            # ksp_idx = np.squeeze(ksp_idx, 0)
            # ksp_idx = torch.from_numpy(ksp_idx)

        ksp_idx = torch.squeeze(ksp_idx, 0) # 12x171x136x156
        # maps = torch.squeeze(maps, 0)
        print("ksp_idx shape:", ksp_idx.shape)
        
        return ksp_idx

    def __len__(self):
        # Total number of slices from all scans
        return int(np.sum(self.num_slices))


# import os
# import scipy.io

# def read_mat_files(directory):
#     for filename in os.listdir(directory):
#         filepath = os.path.join(directory, filename)
#         if os.path.isdir(filepath):
#             # 递归处理子目录
#             read_mat_files(filepath)
#         elif filename.endswith('.mat'):
#             # 处理.mat文件
#             mat = scipy.io.loadmat(filepath)
#             # 对mat数据进行处理
#             # ...

# # 指定根目录路径
# root_dir = '/data'

# # 读取所有.mat文件
# read_mat_files(root_dir)

# 定义一个递归函数，用于遍历目录并获取mat文件路径
def get_mat_paths(dir_path, file_list):
    for filename in os.listdir(dir_path):
        # 获取文件路径
        filepath = os.path.join(dir_path, filename)
        # 判断是否是文件夹，如果是，则递归调用自己
        if os.path.isdir(filepath):
            get_mat_paths(filepath, file_list)
        # 如果是mat文件，则将路径添加到列表中
        elif filename.endswith('.mat'):
            file_list.append(filepath)



class FastMRIv2DataSet(Dataset):
    def __init__(self, config, mode):
        super(FastMRIv2DataSet, self).__init__()
        self.config = config
        if mode == 'train':  
            # self.kspace_dir = '/data1/chentao/data/fastMRI2.0/train_v7.0_mix_Avg_Full/'
            # self.kspace_dir = '/data0/chentao/data/fastMRI2.0/train_v7.0_mix_Avg_Full/'
            self.kspace_dir = '/data0/zhuoxu/data/UIH_data_sc_train/'
        elif mode == 'retro':
            self.kspace_dir = '/data/yj/dataset/retro_v1.0_mix/'
        elif mode == 'sample':
            # self.kspace_dir = '/data1/zhuoxu/data/fastMRI2.0/test_v7.0_mix_selected_simple_PPA_5X_Init_showcase0330/'
            # self.kspace_dir = '/data0/zhuoxu/code/heat_diffusion_1ch/data/dwi_eddy'
            # self.kspace_dir = '/data0/zhuoxu/code/heat_diffusion_1ch/data/test'

            # self.kspace_dir = '/data0/zhuoxu/data/UIH_DWI_Data0829/UID_7320853364705712511_epi_dwi_tra_trig_A4'
            # self.kspace_dir = '/data0/zhuoxu/data/huiren'
            self.kspace_dir = '/data0/zhuoxu/data/huiren/brain/sub_2/3_filtered_nex4'
            
            # self.kspace_dir = '/data0/zhuoxu/files_from_ip49/data/low_field/anke'
            # self.kspace_dir = '/data0/zhuoxu/data/DWI_test/whole_body_batch3_cc12/whole_body_batch3_cc12'
            # self.kspace_dir = '/data0/zhuoxu/data/fastMRI2.0/train_v7.0_mix_Avg_Full/AVG_1/ACS3.0T_data/V4/head/head32/head32_t2_ep_tra_256_full_C2_5/'
            # self.maps_dir = '/data0/zhuoxu/code/ddim_v2/data/maps'
            # self.maps_list = []
            # get_mat_paths(self.maps_dir, self.maps_list)
        elif mode == 'datashift':
            self.kspace_dir = '/data0/jingcheng/data/UIH_Abd/datashift/'
        else:
            raise NotImplementedError

        self.mode = mode
        self.crop_size = config.data.image_size
        self.file_list = []
        get_mat_paths(self.kspace_dir, self.file_list)
        self.num_slices = len(self.file_list)
        print(self.num_slices)


    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file = self.file_list[idx]

        if self.mode == 'sample': # TODO(yeliu)
            # maps_file = self.maps_list[idx]
            # maps = scio.loadmat(maps_file)['s_maps']
            #maps = mat73.loadmat(maps_file)['s_maps']
            #maps = sp.resize(maps, (self.crop_size, self.crop_size))
            #maps = np.expand_dims(maps, 0)
            #maps = crop(maps, self.crop_size, self.crop_size)
            #maps = np.squeeze(maps, 0)
            # try:
            #     ksp = scio.loadmat(data_file)['ksp_cch'] # NCH, NRO, NPE
            #     calib = scio.loadmat(data_file)['calib_cch']
            # except:
            #     ksp = mat73.loadmat(data_file)['ksp_cch']  # NCH, NRO, NPE
            #     calib = mat73.loadmat(data_file)['calib_cch'] 

            try:
                ksp = scio.loadmat(data_file)['ksp'] # NCH, NRO, NPE
                # calib = scio.loadmat(data_file)['calib']
            except:
                ksp = mat73.loadmat(data_file)['ksp']  # NCH, NRO, NPE
                # calib = mat73.loadmat(data_file)['calib'] 
            calib = ksp 
            # ksp = np.transpose(ksp,(2,1,0))
            # calib = np.transpose(calib,(2,1,0))
            ksp = np.expand_dims(ksp,0)
            calib = np.expand_dims(calib,0)
            # minv = np.std(ksp)
            # ksp = ksp / ( minv)

            need_crop = 1
            if need_crop:
                NCH, NRO, NPE = ksp.shape
                # m_pad = np.ones(ksp.shape)
                ksp = sp.resize(ksp, (NCH, 256, 256))
                # ksp = sp.resize(ksp, (NCH, math.ceil(NRO / 64) * 64, math.ceil(NPE / 64) * 64))
                calib = sp.resize(calib, (NCH, 32, 32))
                # m_pad = sp.resize(m_pad, (NCH, math.ceil(NRO / 64-2) * 64, math.ceil(NPE / 64-2) * 64))
                # m_pad = sp.resize(m_pad, (NCH, math.ceil(NRO / 64) * 64, math.ceil(NPE / 64) * 64))
                # ksp = sp.resize(ksp, (NCH, 320, 320))
                # img = sp.resize(img, (NCH, 320, 320))
                # ksp = np.squeeze(FFT2c(np.expand_dims(img, 0)), 0)  
                # ksp = ksp/abs(ksp).max()     

            # img = np.squeeze(IFFT2c(np.expand_dims(ksp, 0)))

            # #img = sp.resize(img, (self.crop_size, self.crop_size))
            # img = np.expand_dims(img, 0)
            # nb, nc, _, _ = img.shape
            # img = pad_or_crop_tensor(img,[nb,nc,256,256])
            # #img = np.expand_dims(img, 0)
            # ksp = FFT2c(img)
            # ksp = np.squeeze(ksp, 0)

            return ksp, calib, os.path.basename(self.file_list[idx])

        else:  # TODO(congcong)
            try:
                img = scio.loadmat(data_file)['img']
            except:
                img = mat73.loadmat(data_file)['img']

            
            #ksp = np.flip(ksp, axis=0)
            # minv = np.std(ksp)
            # ksp = ksp / (minv)

            # img = IFFT2c(np.expand_dims(np.expand_dims(ksp, 0), 0))
            # nb, nc, _, _ = img.shape
            #img = crop(img,256,256)
            # img = pad_or_crop_tensor(img,[nb,nc,256,256])
            # img = sp.resize(img, [nb,nc,384,384])
            img = np.squeeze(img,0)

            #img = sp.resize(img, (self.crop_size, self.crop_size))
            #img = np.expand_dims(img, 0)
            
            return img


    def __len__(self):
        # Total number of slices from all scans
        # return 1 #int(np.sum(self.num_slices)) # TODO: liuye original
        return int(np.sum(self.num_slices))


def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset)

    if config.data.dataset == 'fastMRI':
        dataset = FastMRIKneeDataSet(config, mode)
    elif config.data.dataset == 'single_channel':
        dataset = FastMRIv2DataSet(config, mode)
    elif config.data.dataset == 'fastMRIv2':
        dataset = FastMRIv2DataSet(config, mode)
    elif config.data.dataset == 'MRI_3D':
        dataset = DataSet_3D(config, mode)
    else:
        raise NotImplementedError
    
    if mode == 'train':
        data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
        # with open('/data0/zhuoxu/code/Res_Diffusion2/data_dir/dataloader_module.pkl', 'wb') as file:
        #     dill.dump(data, file)
        # data = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    # test: 90多张图，sample：一张图，第十张
    else:
        data = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=False)


    print(mode, "data loaded")

    return data


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
