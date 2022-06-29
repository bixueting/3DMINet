import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
from glob import glob
from Common import point_operation
import os
warnings.filterwarnings('ignore')
from torchvision import transforms
from Common import data_utils as d_utils
from Common import point_operation
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['19950406']['train']
    return data



point_transform = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
        #d_utils.PointcloudRandomInputDropout(),
    ]
)

point_transform2 = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        #d_utils.PointcloudRandomInputDropout(),
    ]
)

class H5DataLoader(Dataset):
    def __init__(self, opts,augment=False, partition='train'):
        self.opts = opts
        self.num_points = opts.np


        h5_file = '/home/ubuntu/bixueting/code/3DMINet/data/microwavepoint2048.hdf5'
        print("---------------h5_file:",h5_file)
        self.data = load_h5(h5_file)
        self.labels = None

        self.data = self.opts.scale * point_operation.normalize_point_cloud(self.data)
        self.augment = augment
        self.partition = partition

        micropath = '/home/ubuntu/bixueting/code/3DMINet/data/1_micro.npy'
        micro = np.load(micropath)
        microsignal = micro.reshape(5000, 1, 2560)
        self.microsignal = microsignal


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point_set = self.data[index][:self.num_points,:3].copy()
        np.random.shuffle(point_set)
        signalwave = self.microsignal[index].copy()

        if self.augment:
            point_set = point_operation.rotate_point_cloud_and_gt(point_set)
            point_set = point_operation.random_scale_point_cloud_and_gt(point_set)
        point_set = point_set.astype(np.float32)
        signalwave = signalwave.astype(np.float32)

        # if self.con:
        #     label = self.labels[index].copy()
        #     return torch.Tensor(point_set), torch.Tensor(label)
        # return torch.Tensor(point_set), signalwave
        return torch.from_numpy(point_set), signalwave
