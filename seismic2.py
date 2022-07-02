import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from numpy import genfromtxt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from psd_tools import PSDImage


# Seismic data, Sleipner CO2 injections, 2001, 2004 processings (see readme.md for more)
plt.figure()
dat_04 = genfromtxt('data/Seismic/x04_2.txt', delimiter=',')
dat_01 = genfromtxt('data/Seismic/x01_2.txt', delimiter=',')
plt.imshow(dat_04, cmap='seismic')
plt.colorbar()
plt.axis('auto')

# Slice 112 (a singe seismic wave passing threw C02 phase)
s = dat_04[:, 112]
t = np.linspace(0, 1000, 1001)
plt.figure()
plt.plot(t, s)

# Hilbert transform
s_hat = signal.hilbert(s)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t, s_hat.real, s_hat.imag)

# SNR computing functions and 
def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

  
# Gaussian kernel 2D denoising
img_gauss = cv2.GaussianBlur(dat_04, (3, 3), 0, borderType=cv2.BORDER_CONSTANT)
plt.figure()
plt.title('Débruitage par noyau gaussien')
plt.subplot(121)
plt.imshow(dat_04, cmap='seismic')
plt.colorbar()
plt.axis('auto')
plt.subplot(122)
plt.imshow(img_gauss, cmap='seismic')
plt.axis('auto')
plt.colorbar()

plt.figure()
plt.title('Coupe du signal débruité (noyau gaussien)')
plt.subplot(211)
plt.plot(t, s)
plt.subplot(212)
s_gauss = img_gauss[:, 112]
plt.plot(t, s_gauss)

# Total variation denoising (see A. Chambolle article or scikit image documentation for more)
img_TV = denoise_tv_chambolle(dat_04, weight=0.9, eps=0.00002, max_num_iter=10000,
                     multichannel=False, channel_axis=None)
plt.figure()
plt.title('Débruitage par variation totale')
plt.subplot(121)
plt.imshow(dat_01, cmap='seismic')
plt.axis('auto')
plt.subplot(122)
plt.imshow(img_TV, cmap='seismic')
plt.axis('auto')
plt.colorbar()

plt.figure()
plt.title('Coupe du signal débruité (variation totale)')
plt.subplot(211)
plt.plot(t, s)
plt.subplot(212)
s_vt = img_TV[:, 112]
plt.plot(t, s_vt)

# Denoising influence on PSNR and Hilbert transform
psnr_gauss = calculate_psnr(img_gauss, dat_04)
sG_hat = signal.hilbert(s_gauss)
psnr_TV = calculate_psnr(img_TV, dat_04)
w = np.linspace(0, 1, 200)
w = np.delete(w, 0)
psnrTV = np.zeros(len(w))
sTV_hat = np.zeros((len(w), len(t)), dtype=complex)
d_TV = np.zeros(len(w))
d_TV_Lp = np.zeros(len(w))
for k in range(len(w)):
    img_TV_tmp = denoise_tv_chambolle(dat_04, weight=w[k], eps=0.00002,
                                      max_num_iter=10000, multichannel=False,
                                      channel_axis=None)
    psnrTV[k] = calculate_psnr(img_TV_tmp, dat_04)
    sTV_hat[k, :] = signal.hilbert(img_TV_tmp[:, 112])

plt.figure()
plt.title('Dépendance du PSNR au poids de débruitage')
plt.plot(w, psnrTV)
plt.xlabel('Poids')
plt.ylabel('PSNR')

fig = plt.figure()
plt.title('Transformées de Hilbert')
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(t, s_hat.real, s_hat.imag)

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(t, sG_hat.real, sG_hat.imag)

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(t, sTV_hat[180, :].real, sTV_hat[180, :].imag)

plt.figure()
plt.title('Écart moyen des transformées en fonction du poids')
plt.plot(w, dG_av, label='Filtre gaussien')
plt.plot(w, dTV_av, label='Variation totale')
plt.xlabel('Poids w')
plt.ylabel('Écart moyen')
plt.legend()

plt.figure()
plt.title('PSNR en fonction du poids')
tmp = psnr_gauss
psnr_gauss = np.zeros(len(w))
for k in range(len(w)):
    psnr_gauss[k] = tmp
plt.plot(w, psnr_gauss, label='Filtre gaussien')
plt.plot(w, psnrTV, label='Filtre gaussien')
plt.xlabel('Poids w')
plt.ylabel('PSNR')
plt.legend()

# Noise characterization
ft = np.fft.fft2(dat_04)
freq = np.fft.fftfreq(256)
plt.figure()
plt.imshow(np.log10(abs(ft)), cmap='seismic')
plt.axis('auto')
plt.colorbar()

ft2 = np.fft.fft2(img_TV)
freq2 = np.fft.fftfreq(256)
plt.figure()
plt.imshow(np.log10(abs(ft2)), cmap='seismic')
plt.axis('auto')
plt.colorbar()

# Denoised signal amplitude (slice)
amp = np.abs(signal.hilbert(s_vt))
plt.figure()
plt.title('Amplitude du signal')
plt.plot(t, amp)

# Global denoised signal amplitude
t = np.linspace(0, img_TV.shape[1], img_TV.shape[1]+1)
amp_gl = np.abs(signal.hilbert(img_TV))

fig = plt.figure()
plt.imshow(amp_gl, cmap='Spectral')
plt.axis('auto')
plt.colorbar()

plt.show()
