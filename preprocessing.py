import numpy as np
from scipy import signal
from scipy.special import erf
import cv2

pos_0 = 11
ln_t = pos_0 * 2 + 1
sig2 = np.zeros((1, ln_t))
sig2[0, pos_0] = 1
sig2[0, pos_0 + 1] = 2 * 0.305334091
sig2[0, pos_0 - 1] = 2 * 0.305334091
sig2[0, pos_0 + 3] = 2 * -0.072698593
sig2[0, pos_0 - 3] = 2 * -0.072698593
sig2[0, pos_0 + 5] = 2 * 0.021809577
sig2[0, pos_0 - 5] = 2 * 0.021809577
sig2[0, pos_0 + 7] = 2 * -0.005192756
sig2[0, pos_0 - 7] = 2 * -0.005192756
sig2[0, pos_0 + 9] = 2 * 0.000807762
sig2[0, pos_0 - 9] = 2 * 0.000807762
sig2[0, pos_0 + 11] = 2 * -0.000060081
sig2[0, pos_0 - 11] = 2 * -0.000060081
import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    # Python random seed
    random.seed(seed)

    # Numpy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)

    # For GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # All GPUs

    # Ensures the deterministic behavior of convolution operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
def normalize(input_image, max_value):
    input_image = (input_image / max_value) #- 1

    return input_image
def scalevMinvMax(x,vmin, vmax):
    x2 = ((x - vmin)/(vmax - vmin))
    x2[x2<0.] = 0.
    x2[x2>1.] = 1.0
    return x2
def imageRGB_vminvmax(I,vmin=0.0,vmax=0.8):
    """
    imageRGB : processing the image before visualization
    : I : Input satellite image
    : rgbMinMax : Processed image
    """
    rMinMax = scalevMinvMax(I[:,:,2],vmin,vmax)
    gMinMax = scalevMinvMax(I[:,:,1],vmin,vmax)
    bMinMax = scalevMinvMax(I[:,:,0],vmin,vmax)
    rgbMinMax = np.dstack((rMinMax,gMinMax,bMinMax))
    return rgbMinMax
def Convolution_opMS(Image, size, strides):
    """
    Convolution_opMS : a function to divide image into subimages
    :Image : big image
    : size : size of subimages
    : strides : the stride of convolution
    small_images : the input image divided into N subimages
    """
    start_x = 0
    start_y = 0
    end_x = Image.shape[0] - size[0]
    end_y = Image.shape[1] - size[1]

    n_rows = (end_x // strides[0]) + 1
    n_columns = (end_y // strides[1]) + 1
    small_images = []
    for i in range(n_rows):
        for j in range(n_columns):
            new_start_x = start_x + i * strides[0]
            new_start_y = start_y + j * strides[1]
            small_images.append(Image[new_start_x:new_start_x + size[0], new_start_y:new_start_y + size[1], :])
    small_images = np.asanyarray(small_images)
    return small_images
def add_sym(pan, f_sz, n_f=10):
    # return matrix symetrique enleve les commentaires pr compendre
    # pan=np.array([[1,2,3,4],[5,6,7,8],[-1,-2,-3,-4],[-5,-6,-7,-8]])
    # f_sz = 4
    # n_f  = 1
    (ln_x, ln_y) = pan.shape
    ln_a = f_sz * n_f
    ln_xx = ln_x + ln_a * 2
    ln_yy = ln_y + ln_a * 2
    pan_add = np.zeros((ln_xx, ln_yy))
    pan_add[ln_a:ln_a + ln_x, ln_a:(ln_a + ln_y)] = pan

    for i in range(0, n_f):
        pan_add[(ln_a - f_sz - i * f_sz):(ln_a - i * f_sz), (ln_a):(ln_a + ln_y)] = pan[
            (0 + i * f_sz):(f_sz + i * f_sz), :]
        pan_add[(0 + ln_a + ln_x + i * f_sz):(f_sz + ln_a + ln_x + i * f_sz), (ln_a):(ln_a + ln_y)] = pan[
            (0 - f_sz + ln_x - i * f_sz):(f_sz - f_sz + ln_x - i * f_sz), :]
    for i in range(0, n_f):
        pan_add[:, (ln_a - f_sz - i * f_sz):(ln_a - i * f_sz)] = pan_add[
            :, (0 + i * f_sz + ln_a):(f_sz + i * f_sz + ln_a)]
        pan_add[:, (0 + ln_a + ln_y + i * f_sz):(f_sz + ln_a + ln_y + i * f_sz)] = pan_add[
            :, (0 - f_sz + ln_y - i * f_sz + ln_a):(f_sz - f_sz + ln_y - i * f_sz + ln_a)]

    return pan_add

def conv_add_sym(M,PSF,kind):
	f_sz = 4
	n_f  = 10
	l_n = f_sz*n_f
	M_add_sym = add_sym(M,f_sz,n_f)
	#M_add_sym
	#M_add_Blurred
	#start = timer()
	M_add_Blurred = signal.fftconvolve(M_add_sym,PSF,mode='full')
	#print('Conv2d signal full takes=')
	#print(timer() - start)
	#print(M_add_Blurred[149:154,149:154])
	#M_add_Blurred
	(l_r,l_c) = M_add_Blurred.shape
	M_add_Blurred = M_add_Blurred[(l_n):(l_r-l_n),(l_n):(l_c-l_n)]
	return M_add_Blurred

def downsample_MTF(img, val_pos, freq, hsize=33):
    (row, col) = img.shape
    pos = 1 / (2 * freq)  # 1/4 resolution of MS
    # sigma=sqrt(log(val_pos^(-2)))/(pos*2*pi); # ecart-type
    sigma = np.sqrt(np.log(val_pos ** (-2))) / (pos * 2 * np.pi)
    # hsize = 33
    tg = np.arange(-(hsize - 1) / 2, (hsize + 1) / 2)

    #
    gau = np.exp((-tg ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    # c bon
    tg2 = np.arange(1, (hsize + 3) / 2)

    tg2 = tg2 - 0.5

    dgs2 = 0.5 * erf(tg2 / (np.sqrt(2) * sigma))

    # taille diminue par 1
    gs2 = dgs2[1:len(dgs2)] - dgs2[0:len(dgs2) - 1]

    gau2 = np.concatenate((gs2[::-1], [2 * dgs2[0]], gs2))
    # print('fct downsample_MTF using hind method')
    gau = gau2
    gau = gau / gau.sum()
    s = gau.sum()
    gau = np.expand_dims(gau, axis=0)
    PSF = gau.transpose() * gau
    PSF = PSF / PSF.sum()
    pan2m_Blurred = conv_add_sym(img, PSF, 'full')
    xr = int(freq / 2)
    xc = int(freq / 2)
    # on window (1:4,1:4) positionning at pixel (2,2)

    return (pan2m_Blurred[
        (int((hsize + 1) / 2 + xr - 1)):(int((hsize + 1) / 2 + row - 1 + xr)):freq, int((hsize + 1) / 2 + xc - 1):int(
            ((hsize + 1) / 2 + col - 1 + xc)):freq])
def generate_filter(sig_in,sig2_in,ratio):
	ln_t_in = sig_in.shape[1]
	ln_t_in2 = sig2_in.shape[1]
	pos_0_in = int((ln_t_in-1)/2)
	pos_0_in2 = int((ln_t_in2-1)/2)
	pos_0 = pos_0_in2 + int(ratio/2) * pos_0_in
	ln_t = pos_0 * 2 + 1
	sig2_new =  np.zeros((1,ln_t))
	sig_in_new = np.zeros((1, ln_t))
	sig12 = np.zeros((1, ln_t))
	sig2_new[0,pos_0-int(sig2_in.shape[1]/2): pos_0+int(sig2_in.shape[1]/2)+1] = sig2_in
	sig_in_new[0,pos_0-int(sig_in.shape[1]/2): pos_0+int(sig_in.shape[1]/2)+1] = sig_in

	for i in range(-pos_0,pos_0+1):
		posi = pos_0+i
		val=0
		for j in range(-11,12):
			posj=pos_0+j
			if ((posi-ratio/2*j+1) <= ln_t) and ((posi-ratio/2*j+1) > 0):
				val = val+sig_in_new[0,posj]*sig2_new[0,int(posi-ratio/2*j)]
		sig12[0,posi] = val
	return sig12
def upsample_perfect(pan2m_mft,freq_up=4):
	#pan2m_mft = [[101,103,105],[111,113,115],[121,123,125]]
	#pan2m_mft=np.asarray([[0,1,2,3,10,11,12,13],[0,1,2,3,10,11,12,13],[0,1,2,3,10,11,12,13],[0,1,2,3,10,11,12,13]])
	pan2m_mft=np.array(pan2m_mft)
	f_sz = 10
	nf = 1
	pan05_perfect_padded = add_sym(pan2m_mft, f_sz ,nf)

	(l_r,l_c)=pan05_perfect_padded.shape
	row = freq_up*l_r
	col = freq_up*l_c

	# ln_t = 67
	pos_0 = 33
	# ln_t = pos_0 *2 +1
	sig_freq_up = generate_filter(sig2, sig2,4)
	kernel = np.transpose(sig_freq_up)*sig_freq_up
	pan05_mft_up = np.zeros((row,col))
	mid_pixel = int(freq_up/2)-1
	pan05_mft_up[mid_pixel:-1:freq_up,mid_pixel:-1:freq_up] = pan05_perfect_padded
	pan05_mft_up_fil = signal.fftconvolve(pan05_mft_up,kernel,mode='same')

	pan05_mft_up_fil_unpadded = pan05_mft_up_fil[freq_up*f_sz*nf:pan05_mft_up_fil.shape[0]-freq_up*f_sz*nf,freq_up*f_sz*nf:pan05_mft_up_fil.shape[1]-freq_up*f_sz*nf]
	return pan05_mft_up_fil_unpadded

def down_up_sentinel_4(img, freq_down_up = 4):
    img_down_up= np.zeros_like(img[:10,:,:])
    for b in range(10):
        band = img[b, :, :]
        dsize = (int(band.shape[0] / freq_down_up) * freq_down_up, int(band.shape[1] / freq_down_up) * freq_down_up)
        band4 = cv2.resize(band, dsize)
        band4_down = downsample_MTF(band4, 0.2, freq_down_up, 171)
        band4_up = upsample_perfect(band4_down, freq_down_up)
        band4_up = cv2.resize(band4_up, (band.shape[1], band.shape[0]))
        img_down_up[b,:,:] = band4_up
    return img_down_up


