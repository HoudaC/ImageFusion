
import matplotlib.pyplot as plt
from preprocessing import imageRGB_vminvmax
import numpy as np
vmin =0.0
vmax=0.8

# Function to plot images and their spectra
def plot_images(lr_image, gt_hr_image, sr_image, cond_sr_image=None):

    if cond_sr_image is  None:
        # Plot Original HR Image, Generated SR Image, and Low-Resolution Image
        fig, axs = plt.subplots(1, 3)

        # Original HR Image
        axs[1].imshow(imageRGB_vminvmax(gt_hr_image, vmin, vmax))
        axs[1].set_title('HR GT Image')
        axs[1].axis('off')

        # Generated SR Image
        axs[2].imshow(imageRGB_vminvmax(sr_image, vmin, vmax))
        axs[2].set_title('Generated SR Image (SRCNN)')
        axs[2].axis('off')

        # Low-Resolution Image
        axs[0].imshow(imageRGB_vminvmax(lr_image, vmin, vmax))
        axs[0].set_title('Low-Resolution Image')
        axs[0].axis('off')


        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(2, 2)
        # Low-Resolution Image
        axs[0,0].imshow(imageRGB_vminvmax(lr_image, vmin, vmax))
        axs[0,0].set_title('Low-Resolution Image')
        axs[0,0].axis('off')


        # Original HR Image
        axs[0,1].imshow(imageRGB_vminvmax(gt_hr_image, vmin, vmax))
        axs[0,1].set_title('HR GT Image')
        axs[0,1].axis('off')

        # Generated SR Image
        axs[1,0].imshow(imageRGB_vminvmax(sr_image, vmin, vmax))
        axs[1,0].set_title('Generated SR Image (SRCNN)')
        axs[1,0].axis('off')

        # Generated SR Image
        axs[1,1].imshow(imageRGB_vminvmax(cond_sr_image, vmin, vmax))
        axs[1,1].set_title('Generated SR Image (Guided SRCNN)')
        axs[1,1].axis('off')



        plt.tight_layout()
        plt.show()

def evaluate_spectral_fidelity(lr_image, gt_hr_image, sr_image, cond_sr_image=None):

    sentinel_wavelength = [489, 559, 665, 703, 740, 783, 850, 864, 1610, 2192]
    plt.figure()

    # Frequency Spectrum for Original HR Image
    plt.plot(sentinel_wavelength,np.mean(sr_image, axis=(0, 1)), "ro", label="Generated SR (SRCNN)")
    plt.plot(sentinel_wavelength, np.mean(lr_image, axis=(0, 1)), "gs", label="Low resolution")
    plt.plot(sentinel_wavelength,np.mean(gt_hr_image, axis =(0,1)), "b*", label="GT")

    plt.title('Reflectance')
    if cond_sr_image is not None:
        plt.plot(sentinel_wavelength, np.mean(cond_sr_image, axis=(0, 1)), "m^", label="Generated SR (Guided SRCNN)")


    plt.legend(fontsize=10, loc="lower center")


    plt.tight_layout()
    plt.show()