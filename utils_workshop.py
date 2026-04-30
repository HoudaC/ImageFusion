import cv2
import rasterio
import numpy as np
import os
import torch
from metrics import calculate_psnr, calculate_ssim, calculate_rmse, plot_images_and_spectra
from preprocessing import down_up_sentinel_4
def normalize(input_image, max_value):
    input_image = (input_image / max_value) #- 1
    return input_image

def read_data2test(data_test_dir, only_sentinel= True):
    lst = os.listdir(data_test_dir)
    img_all = []
    img_names = []
    i = 0
    for dir in lst:
        ## reading sentinel imahe
        hr_sentinel_img_path = os.path.join(data_test_dir, dir, "hr_sentinel_" + dir + ".tif" )
        with rasterio.open(hr_sentinel_img_path) as src:
            hr_sentinel_img = src.read()
        hr_sentinel_img_normalized = normalize(hr_sentinel_img, 10000)
        hr_sentinel_img_normalized = np.float32(hr_sentinel_img_normalized)

        hr_cond_img_path = os.path.join(data_test_dir, dir,  "hr_guide_rgbnir_" + dir + ".tif")
        with rasterio.open(hr_cond_img_path) as src:
            hr_cond_img = src.read()
        hr_cond_img_normalized = normalize(hr_cond_img, 10000)
        hr_cond_img_normalized = np.float32(hr_cond_img_normalized)

        if only_sentinel:
            img_normalized = hr_sentinel_img_normalized
        else:
            img_normalized = np.concatenate([hr_sentinel_img_normalized,hr_cond_img_normalized])

        img_all.append(img_normalized)
        img_names.append(dir)


    return img_all, img_names


def apply_bicubic_interpolation(in_img):
    freq_down = 4
    img = down_up_sentinel_4(in_img[0])
    img_bicubic = np.zeros_like(img)
    for b in range(10):
        band = img[b, :, :]
        dsize = (int(band.shape[0] / freq_down) , int(band.shape[1] / freq_down))
        band4 = cv2.resize(band, dsize)
        band_up = cv2.resize(band4, (band.shape[1], band.shape[0]), interpolation=cv2.INTER_CUBIC)
        img_bicubic[b, :, :] = band_up
    return img.transpose(1,2,0), img_bicubic.transpose(1,2,0)

# Test function to evaluate the model
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    lr_images_patches = []
    hr_images_patches = []
    sr_images_patches = []

    with torch.no_grad():
        for batch_idx, (lr_images, hr_images) in enumerate(test_loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            # Get predictions (upscaled LR images)
            sr_images = model(lr_images)


            # FIX: clamp in torch BEFORE numpy
            sr_images = torch.clamp(sr_images, 0, 1)
            hr_images = torch.clamp(hr_images, 0, 1)

            # Convert images from Torch Tensor to numpy
            lr_images = lr_images.cpu().numpy()
            sr_images = sr_images.cpu().numpy()
            hr_images = hr_images.cpu().numpy()

            lr_images_patches.append(lr_images[0])
            hr_images_patches.append(hr_images[0])
            sr_images_patches.append(sr_images[0])

    return np.array(lr_images_patches), np.array(hr_images_patches), np.array(sr_images_patches)

# Test function to evaluate the model
def conditionnal_test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    lr_images_patches = []
    hr_images_patches = []
    sr_images_patches = []

    with torch.no_grad():
        for batch_idx, (lr_images,conditional_images, hr_images) in enumerate(test_loader):
            lr_images,conditional_images,  hr_images = lr_images.to(device),conditional_images.to(device),hr_images.to(device)

            # Get predictions (upscaled LR images)
            sr_images = model(lr_images, conditional_images)

            sr_images = torch.clamp(sr_images, 0, 1)
            hr_images = torch.clamp(hr_images, 0, 1)
            lr_images = lr_images.cpu().numpy()
            sr_images = sr_images.cpu().numpy()
            hr_images = hr_images.cpu().numpy()

            lr_images_patches.append(lr_images[0])
            hr_images_patches.append(hr_images[0])
            sr_images_patches.append(sr_images[0])


    return  np.array(lr_images_patches), np.array(hr_images_patches), np.array(sr_images_patches)



