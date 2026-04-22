from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from math import sqrt, log10
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import imageRGB_vminvmax

# Define the PSNR and SSIM functions

def calculate_psnr(hr_image, sr_image):
    mse = np.mean((hr_image - sr_image) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0  # Assuming normalized images between 0 and 1
    return 20 * log10(pixel_max / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculates Structural Similarity (SSIM) between two images."""
    # img1 = img1.cpu().numpy()
    # img2 = img2.cpu().numpy()
    return ssim(img1, img2, multichannel=True)
# Function to compute RMSE
def calculate_rmse(hr_image, sr_image):
    return sqrt(mse(hr_image.flatten(), sr_image.flatten()))

vmin =0.0
vmax=0.5
# Function to compute the frequency spectrum of an image
def compute_spectrum(image):
    # Perform FFT and shift the zero-frequency component to the center
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    # Apply logarithmic scaling for better visualization
    magnitude_spectrum = np.log(1 + magnitude_spectrum)
    return magnitude_spectrum

# Function to plot images and their spectra
def plot_images_and_spectra(lr_image, hr_image, sr_image, image_title):
    # Plot Original HR Image, Generated SR Image, and Low-Resolution Image
    fig, axs = plt.subplots(1, 4)

    # Original HR Image
    axs[0].imshow(imageRGB_vminvmax(hr_image, vmin, vmax))
    axs[0].set_title('Original HR Image')
    axs[0].axis('off')

    # Generated SR Image
    axs[1].imshow(imageRGB_vminvmax(sr_image, vmin, vmax))
    axs[1].set_title('Generated SR Image')
    axs[1].axis('off')

    # Low-Resolution Image
    axs[2].imshow(imageRGB_vminvmax(lr_image, vmin, vmax))
    axs[2].set_title('Low-Resolution Image')
    axs[2].axis('off')
    sentinel_wavelength = [489, 559, 665, 703, 740, 783, 850, 864, 1610, 2192]

    # Frequency Spectrum for Original HR Image
    axs[3].plot(sentinel_wavelength,np.mean(hr_image, axis =(0,1)), "b*", label="GT")
    axs[3].plot(sentinel_wavelength,np.mean(sr_image, axis=(0, 1)), "r*", label="generaed superresoltiont")
    axs[3].plot(sentinel_wavelength, np.mean(lr_image, axis=(0, 1)), "g*", label="Low resolution")
    axs[3].set_title('Spectrum')
    axs[3].legend(fontsize=10, loc="lower center")
    plt.suptitle(image_title)

    plt.tight_layout()
    plt.show()

