import cv2
import matplotlib.pyplot as plt


import torch
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from preprocessing import Convolution_opMS, imageRGB_vminvmax, set_seed
from utils import define_stride, reconstruct_image_avg
from SRCNN_model import SRCNN, ConditionalSRCNN
from metrics import calculate_psnr, calculate_ssim, calculate_rmse, plot_images_and_spectra , plot_images_and_compare

from generate_dataloader import sr_dataloader
from utils_workshop import read_data2test, test,conditionnal_test, apply_bicubic_interpolation

if __name__=="__main__":
    set_seed(42)
    # Assuming `model` is your trained SRCNN model and it’s on the appropriate device (e.g., CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)  # Make sure to load your trained model

    # Load the trained model weights
    model.load_state_dict(torch.load('./weights/best_model.pth'))  # Change the path if necessary

    cond_model = ConditionalSRCNN().to(device)  # Make sure to load your trained model

    # Check if checkpoint exists and load it

    cond_model.load_state_dict(torch.load("weights_cond/best_model_cond.pth"))

    # Run the test function to
    data_test_dir = "./Dataset/testing_dataset4workshop"
    img_size = 32
    test_dataloader_batch_size = 1
    patch = (img_size, img_size)
    img_test_all, img_test_names = read_data2test(data_test_dir, only_sentinel=False)
    img_test_all = np.stack(img_test_all, axis=0)
    img = img_test_all[0]
    strides = define_stride(img.shape[1], img_size)[0], define_stride(img.shape[2], img_size)[0]
    original_shape = img.transpose(1, 2, 0)[:, :, :10].shape

    # img_totest_idx =40
    for idx in range(img_test_all.shape[0]):
        img_test = np.array([img_test_all[idx]])
        img_lr, sr_bicubic = apply_bicubic_interpolation(img_test[:,:10,:,:])

        test_dataloader = sr_dataloader(img_test[:,:10,:,:], patch, strides, test_dataloader_batch_size)

        avg_psnr, avg_ssim, avg_rmse, lr_patches, hr_patches, sr_patches = test(model, test_dataloader, device)


        hr_image = reconstruct_image_avg(hr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)
        lr_image = reconstruct_image_avg(lr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)
        sr_image = reconstruct_image_avg(sr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)

        plot_images_and_compare(lr_image, hr_image, sr_image, sr_bicubic)
        rmse_bicubic = calculate_rmse(hr_image, sr_bicubic)
        psnr_bicubic = calculate_psnr(hr_image, sr_bicubic)
        ssim_bicubic = calculate_ssim(hr_image, sr_bicubic)
        print("Bicubic interrpolation      ssim=", ssim_bicubic, "    psnr=", psnr_bicubic, "       rmse=", rmse_bicubic)


        rmse = calculate_rmse(hr_image, sr_image)
        psnr = calculate_psnr(hr_image, sr_image)
        ssim = calculate_ssim(hr_image, sr_image)
        print("SRCNN       ssim=", ssim, "    psnr=", psnr, "       rmse=", rmse)

        plot_images_and_spectra(lr_image, hr_image, sr_image, "Image number    "+  str(idx))

        cond_test_dataloader = sr_dataloader(img_test, patch, strides, test_dataloader_batch_size, only_sentinel=False)

        cond_avg_psnr, cond_avg_ssim, cond_avg_rmse, cond_lr_patches, cond_hr_patches, cond_sr_patches = conditionnal_test(cond_model, cond_test_dataloader, device)

        cond_sr_image = reconstruct_image_avg(cond_sr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)
        cond_rmse = calculate_rmse(hr_image, cond_sr_image)
        cond_psnr = calculate_psnr(hr_image, cond_sr_image)
        cond_ssim = calculate_ssim(hr_image, cond_sr_image)
        print("Guided SRCNN       ssim=", cond_ssim, "    psnr=", cond_psnr, "       rmse=", cond_rmse)
        plot_images_and_spectra(lr_image, hr_image, cond_sr_image, "Image number    "+  str(idx))



