import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.fftpack
import scipy.io as sio
import copy
import pylab as pl
import time
from IPython import display

## create signal
srate = 1000 # Hz
time  = np.arange(0,3,1/srate)
n     = len(time)
p     = 15 # poles for random interpolation

# noise level, measured in standard deviations
noiseamp = 5

# amplitude modulator and noise level
ampl   = np.interp(np.linspace(0,p,n),np.arange(0,p),np.random.rand(p)*30)
noise  = noiseamp * np.random.randn(n)
signal1= ampl + noise

# subtract mean to eliminate DC
signal1 = signal1 - np.mean(signal1)


## create Gaussian spectral shape
# Gaussian parameters (in Hz)
peakf = 11
fwhm  = 5.2

# vector of frequencies
hz = np.linspace(0,srate,n)

# frequency-domain Gaussian
s  = fwhm*(2*np.pi-1)/(4*np.pi)  # normalized width
x  = hz-peakf              # shifted frequencies
fx = np.exp(-.5*(x/s)**2)     # gaussian


## now for convolution

# FFTs
dataX = scipy.fftpack.fft(signal1)

# IFFT
convres = 2*np.real( scipy.fftpack.ifft( dataX*fx ))

# frequencies vector
hz = np.linspace(0,srate,n)


### time-domain plot

# lines
plt.plot(time,signal1,'r',label='Signal')
plt.plot(time,convres,'k',label='Smoothed')
plt.xlabel('Time (s)'), plt.ylabel('amp. (a.u.)')
plt.legend()
plt.title('Narrowband filter')
plt.show()



### frequency-domain plot

# plot Gaussian kernel
plt.figure()
plt.plot(hz,fx,'k')
plt.xlim([0,30])
plt.ylabel('Gain')
plt.title('Frequency-domain Gaussian')
plt.show()

# raw and filtered data spectra
plt.figure()
plt.plot(hz,np.abs(dataX)**2,'rs-',label='Signal')
plt.plot(hz,np.abs(dataX*fx)**2,'bo-',label='Conv. result')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
plt.legend()
plt.title('Frequency domain')
plt.xlim([0,25])
plt.ylim([0,1e6])
plt.show()