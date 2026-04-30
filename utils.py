import rasterio
import numpy as np
import os
import cv2
import math
from preprocessing import normalize, downsample_MTF
def define_stride(shape,size):
    if math.ceil(shape / size) == 2 :
        stride = shape%size
        new_dim = shape
    else :
        # stride = size
        # new_dim = shape + size - (shape % size)

        num_patches = math.ceil(shape / size)
        stride = (shape - size) // (num_patches - 1)
        new_dim = shape

    return stride,new_dim
def down_img(img, freq_down = 4):
    # row = int(img.shape[1] / freq_down_up) * freq_down_up
    # col = int(img.shape[2] / freq_down_up) * freq_down_up
    #
    # img_down_up = np.zeros([10,row, col])
    img_down = []
    for b in range(img.shape[0]):
        band = img[b, :, :]

        dsize = (int(band.shape[1] / freq_down) * freq_down, int(band.shape[0] / freq_down) * freq_down)
        band_ = cv2.resize(band, dsize)
        band_down = downsample_MTF(band_, 0.2, freq_down, 171)
        img_down.append( band_down)
    img_down = np.array(img_down)
    return img_down
def trainPrepare(img_train_all,patch,stride):

    Pleaide_train = img_train_all[:,10:,:,:]
    Sentinel_train_62cm = img_train_all[:,0:10,:,:]
    Sentinel_train = np.zeros_like(Sentinel_train_62cm)
    for i in range(Sentinel_train_62cm.shape[0]):
        img_down_up = down_up_sentinel_62cm(Sentinel_train_62cm[i,:,:,:])
        Sentinel_train[i,:,:,:] = img_down_up

    Sentinel_train_10m = np.concatenate((Sentinel_train,Pleaide_train), axis=1)
    X_train_list = []
    for img in Sentinel_train_10m:
        Sentinel_train_10m_img = reshape_as_image(img)
        X_train_img = Convolution_opMS(Sentinel_train_10m_img,patch,stride)
        X_train_list.extend(X_train_img)
    X_train = np.array(X_train_list)

    Y_train_list = []
    for img in Sentinel_train_62cm:
        sentinel62_gt_img = reshape_as_image(img)
        Y_train_img = Convolution_opMS(sentinel62_gt_img,patch,stride)
        Y_train_list.extend(Y_train_img)
    Y_train = np.array(Y_train_list)

    return X_train,Y_train

def read_data(data_train_dir,data_size = 1000, only_sentinel= True):
    lst = os.listdir(data_train_dir)
    img_train_all = []
    img_train_names = []
    i = 0
    for dir in lst:
        img_path = os.path.join(data_train_dir, dir )
        # print(img_path)
        with rasterio.open(img_path) as src:
            img_train = src.read()
        img_train_normalized = normalize(img_train, 10000)
        img_train_normalized = np.float32(img_train_normalized)
        if only_sentinel:
            img_train_normalized =img_train_normalized[:10,:,:]

        img_train_all.append(img_train_normalized)
        img_train_names.append(dir)
        i = i + 1
        if i == data_size:
            break
    return img_train_all, img_train_names

def read_generate_data(data_dir, data_save_dir):
    lst = os.listdir(data_dir)
    img_all = []
    i = 0
    for dir in lst:
        img_path = os.path.join(data_dir, dir, dir + ".tif")
        with rasterio.open(img_path) as src:
            img = src.read()
            img_meta = src.meta

            img_down_4 = down_img(img,4)
            img_down_4_metadata = img_meta.copy()
            img_down_4_metadata.update({
                "width": img_down_4.shape[2],
                "height": img_down_4.shape[1]
            })

        print(img_meta)
        print(img_down_4_metadata)
        # savedirectory = data_save_dir + dir
        # if os.path.exists(savedirectory):
        #     shutil.rmtree(savedirectory)
        # os.makedirs(savedirectory)

        imagename = data_save_dir + "/" + dir + ".tif"
        print(imagename)

        with rasterio.open(imagename, "w", **img_down_4_metadata) as dst:
            dst.write(img_down_4)

def reconstruct_image_avg(small_images, original_shape, patch_size, strides):
    size= (patch_size,patch_size)
    H, W, C = original_shape

    full = np.zeros((H, W, C), dtype=np.float32)
    count = np.zeros((H, W, C), dtype=np.float32)

    n_rows = (H - size[0]) // strides[0] + 1
    n_cols = (W - size[1]) // strides[1] + 1

    index = 0
    for i in range(n_rows):
        for j in range(n_cols):
            x = i * strides[0]
            y = j * strides[1]

            full[x:x+size[0], y:y+size[1], :] += small_images[index]
            count[x:x+size[0], y:y+size[1], :] += 1

            index += 1

    return full / count


