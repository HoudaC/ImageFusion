import cv2
import numpy as np
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset, DataLoader
import albumentations as A


from preprocessing import Convolution_opMS, down_up_sentinel_4

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets  # torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        #######################
        #######################
        if (self.transform):
            x = x.transpose(1, 2, 0)
            y = y.transpose(1, 2, 0)
            transformed = self.transform(image=x, image1=y)
            x = transformed['image']
            y = transformed['image1']
            #########################
            x = x.transpose(2, 0, 1)
            y = y.transpose(2, 0, 1)
        # put them as a list
        return x, y

    def __len__(self):
        # return len(self.data)
        return self.data.shape[0]


class MyDataset_cond(Dataset):
    def __init__(self, data, data1, targets, transform=None):
        self.data = data
        self.data1 = data1
        self.targets = targets  # torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        x1 = self.data1[index]
        y = self.targets[index]
        #######################
        #######################
        if (self.transform):
            x = x.transpose(1, 2, 0)
            x1 = x1.transpose(1, 2, 0)
            y = y.transpose(1, 2, 0)
            transformed = self.transform(image=x, image0=x1, image1=y)
            x = transformed['image']
            x1 = transformed['image0']
            y = transformed['image1']
            #########################
            x = x.transpose(2, 0, 1)
            x1 = x1.transpose(2, 0, 1)
            y = y.transpose(2, 0, 1)
        # put them as a list
        return x, x1, y

    def __len__(self):
        # return len(self.data)
        return self.data.shape[0]
def trainPrepare(img_train_all,patch,stride):

    Sentinel_train_sr = img_train_all[:,0:10,:,:]
    Sentinel_train_lr = np.zeros_like(Sentinel_train_sr)
    for i in range(Sentinel_train_sr.shape[0]):
        img_down_up = down_up_sentinel_4(Sentinel_train_sr[i,:,:,:])
        Sentinel_train_lr[i,:,:,:] = img_down_up

    X_train_list = []
    for img in Sentinel_train_lr:
        Sentinel_train_lr_img = reshape_as_image(img)
        X_train_img = Convolution_opMS(Sentinel_train_lr_img,patch,stride)
        X_train_list.extend(X_train_img)
    X_train = np.array(X_train_list)

    Y_train_list = []
    for img in Sentinel_train_sr:
        sentinel4_gt_img = reshape_as_image(img)
        Y_train_img = Convolution_opMS(sentinel4_gt_img,patch,stride)
        Y_train_list.extend(Y_train_img)
    Y_train = np.array(Y_train_list)
    return X_train,Y_train

def trainPrepare_condionnal(img_train_all,patch,stride):
    sensor_train_sr = img_train_all[:,10:,:,:]
    Sentinel_train_sr = img_train_all[:,0:10,:,:]
    Sentinel_train_lr = np.zeros_like(Sentinel_train_sr)
    for i in range(Sentinel_train_sr.shape[0]):
        img_down_up = down_up_sentinel_4(Sentinel_train_sr[i,:,:,:])
        Sentinel_train_lr[i,:,:,:] = img_down_up
    Sentinel_train_10m = np.concatenate((Sentinel_train_lr, sensor_train_sr), axis=1)
    X_train_list = []
    for img in Sentinel_train_10m:
        Sentinel_train_10m_img = reshape_as_image(img)
        X_train_img = Convolution_opMS(Sentinel_train_10m_img,patch,stride)
        X_train_list.extend(X_train_img)
    X_train = np.array(X_train_list)

    Y_train_list = []
    for img in Sentinel_train_sr:
        sentinel4_gt_img = reshape_as_image(img)
        Y_train_img = Convolution_opMS(sentinel4_gt_img,patch,stride)
        Y_train_list.extend(Y_train_img)
    Y_train = np.array(Y_train_list)
    return X_train,Y_train


def sr_dataloader(img_list,patch, strides,batchsize, only_sentinel= True ,applytransform= False):

    transform = A.Compose(
        [A.HorizontalFlip(p=0.2), A.VerticalFlip(p=0.2), A.RandomCrop(height=patch[0], width=patch[1], p=1),
         A.Rotate(border_mode=cv2.BORDER_CONSTANT, p=0.3)],
        additional_targets={'image0': 'image', 'image1': 'image'})

    if only_sentinel:
        X_, Y_ = trainPrepare(img_list, patch, strides)
        X_ = X_.transpose(0, 3, 1, 2)
        Y_ = Y_.transpose(0, 3, 1, 2)
        if applytransform:
            mydataset = MyDataset(X_[:, 0:10, :, :], Y_, transform=transform)
        else:
            mydataset = MyDataset(X_[:, 0:10, :, :], Y_)
    else:
        X_, Y_ = trainPrepare_condionnal(img_list, patch, strides)
        X_ = X_.transpose(0, 3, 1, 2)
        Y_ = Y_.transpose(0, 3, 1, 2)
        if applytransform:
            mydataset = MyDataset_cond(X_[:, 0:10, :, :], X_[:, 10:14, :, :],Y_, transform=transform)
        else:
            mydataset = MyDataset_cond(X_[:, 0:10, :, :], X_[:, 10:14, :, :], Y_)
    dataloader = DataLoader(mydataset, batch_size=batchsize)
    return dataloader


def sr_dataloader_test(images, patch_size, stride, batch_size, only_sentinel=False, applytransform=False):
    patches = []
    patch_coords = []  # This will hold (i, j) coordinates

    for img in images:
        h, w, c = img.shape
        for i in range(0, h - patch_size[0] + 1, stride[0]):
            for j in range(0, w - patch_size[1] + 1, stride[1]):
                patch = img[i:i + patch_size[0], j:j + patch_size[1], :]
                patches.append(patch)
                patch_coords.append((i, j))  # Store patch position

    return patches, patch_coords