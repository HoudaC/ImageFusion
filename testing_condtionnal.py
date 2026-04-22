

import torch
from utils import define_stride, read_data, reconstruct_image_avg
from SRCNN_model import ConditionalSRCNN
from metrics import calculate_psnr, calculate_ssim, calculate_rmse, plot_images_and_spectra

from generate_dataloader import sr_dataloader


import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import set_seed
set_seed(42)

# Test function to evaluate the model
def conditionnal_test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_images = 0
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



            # Calculate PSNR and SSIM for each image in the batch
            for i in range(lr_images.shape[0]):
                rmse = calculate_rmse(hr_images[i], sr_images[i])
                psnr = calculate_psnr(hr_images[i], sr_images[i])
                ssim = calculate_ssim(hr_images[i], sr_images[i])
                total_rmse += rmse
                total_psnr += psnr
                total_ssim += ssim
                # print("ssim=", ssim , "    psnr=", psnr, "       rmse=", rmse)
                # Visualize the original HR, LR, and SR images along with their frequency spectra
                # plot_images_and_spectra(lr_images[i].transpose(1, 2, 0),  # Convert to HWC
                #                         hr_images[i].transpose(1, 2, 0),  # Convert to HWC
                #                         sr_images[i].transpose(1, 2, 0))  # Convert to HWC

                total_images += 1

    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_rmse = total_rmse /total_images
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")

    return avg_psnr, avg_ssim,avg_rmse, np.array(lr_images_patches), np.array(hr_images_patches), np.array(sr_images_patches)



if __name__=="__main__":
    set_seed(42)

    # Assuming `model` is your trained SRCNN model and it’s on the appropriate device (e.g., CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalSRCNN().to(device)  # Make sure to load your trained model

    # Check if checkpoint exists and load it

    model.load_state_dict(torch.load("weights_cond/best_model_cond.pth"))
    data_test_dir = "./Dataset/test0"
    img_size = 32
    test_dataloader_batch_size = 1
    patch = (img_size, img_size)

    img_test_all_, img_test_names = read_data(data_test_dir, 100, only_sentinel=False)
    img_test_all_ = np.stack(img_test_all_, axis=0)
    # img_totest_idx =40
    for img_totest_idx in range(img_test_all_.shape[0]):
        img_test_all =np.array([img_test_all_[img_totest_idx]])

        strides = define_stride(img_test_all[0].shape[1], img_size)[0], define_stride(img_test_all[0].shape[2], img_size)[
            0]
        test_dataloader = sr_dataloader(img_test_all, patch, strides, test_dataloader_batch_size, only_sentinel=False)




        # Run the test function to evaluate the model's performance
        avg_psnr, avg_ssim, avg_rmse, lr_patches, hr_patches, sr_patches = conditionnal_test(model, test_dataloader, device)
        w, h = define_stride(img_test_all.shape[2], img_size)[1], define_stride(img_test_all.shape[3], img_size)[1]
        original_shape = img_test_all[0].transpose(1,2,0)[:,:,:10].shape

        hr_image = reconstruct_image_avg(hr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)

        lr_image = reconstruct_image_avg(lr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)


        sr_image = reconstruct_image_avg(sr_patches.transpose(0, 2, 3, 1), original_shape, patch, strides)
        plot_images_and_spectra(lr_image, hr_image, sr_image, "Image number    "+  str(img_totest_idx))
        rmse = calculate_rmse(hr_image, sr_image)
        psnr = calculate_psnr(hr_image, sr_image)
        ssim = calculate_ssim(hr_image, sr_image)
        print("Full Image" + img_test_names[img_totest_idx] + "    index=", img_totest_idx,"      ssim=", ssim, "    psnr=", psnr, "       rmse=", rmse)
