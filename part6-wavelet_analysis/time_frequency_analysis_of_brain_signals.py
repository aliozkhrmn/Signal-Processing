import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy

# load in data
braindat = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\wavelets\data4TF.mat')
timevec = braindat['timevec'][0]
srate = braindat['srate'][0]
data = braindat['data'][0]

# plot the signal
plt.plot(timevec,data)
plt.xlabel('Time (s)'), plt.ylabel('Voltage (\muV)')
plt.title('Time-domain signal')
plt.show()

###############################

## create complex Morlet wavelets

# wavelet parameters
nfrex = 50  # 50 frequencies
frex = np.linspace(8, 70, nfrex)
fwhm = .2  # full-width at half-maximum in seconds

# time vector for wavelets
wavetime = np.arange(-2, 2, 1 / srate)

# initialize matrices for wavelets
wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

# create complex Morlet wavelet family
for wi in range(0, nfrex):
    # Gaussian
    gaussian = np.exp(-(4 * np.log(2) * wavetime ** 2) / fwhm ** 2)

    # complex Morlet wavelet
    wavelets[wi, :] = np.exp(1j * 2 * np.pi * frex[wi] * wavetime) * gaussian

# show the wavelets
plt.figure()
plt.subplot(121)
plt.plot(wavetime, np.real(wavelets[10, :]), label='Real part')
plt.plot(wavetime, np.imag(wavelets[10, :]), label='Imag part')
plt.xlabel('Time')
plt.xlim([-.5, .5])
plt.legend()
plt.show()

plt.subplot(122)
plt.pcolormesh(wavetime, frex, np.real(wavelets))
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Real part of wavelets')
plt.xlim([-.5, .5])
plt.show()


#####################################


## run convolution using spectral multiplication

# convolution parameters
nconv = len(timevec) + len(wavetime) - 1  # M+N-1
halfk = int(np.floor(len(wavetime) / 2))

# Fourier spectrum of the signal
dataX = scipy.fftpack.fft(data, nconv)

# initialize time-frequency matrix
tf = np.zeros((nfrex, len(timevec)))

# convolution per frequency
for fi in range(0, nfrex):
    # FFT of the wavelet
    waveX = scipy.fftpack.fft(wavelets[fi, :], nconv)
    # amplitude-normalize the wavelet
    waveX = waveX / np.max(waveX)

    # convolution
    convres = scipy.fftpack.ifft(waveX * dataX)
    # trim the "wings"
    convres = convres[halfk - 1:-halfk]

    # extract power from complex signal
    tf[fi, :] = np.abs(convres) ** 2


## plot the results
plt.figure()
plt.pcolormesh(timevec,frex,tf,vmin=0,vmax=1e3)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()