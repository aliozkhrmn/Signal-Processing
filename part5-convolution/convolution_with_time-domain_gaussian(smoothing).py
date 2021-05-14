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


# simulation parameters
srate = 1000 # Hz
time  = np.arange(0,3,1/srate)
n     = len(time)
p     = 15 # poles for random interpolation


## create signal
# noise level, measured in standard deviations
noiseamp = 5

# amplitude modulator and noise level
ampl    = np.interp(np.linspace(0,p,n),np.arange(0,p),np.random.rand(p)*30)
noise   = noiseamp * np.random.randn(n)
signal1 = ampl + noise

# subtract mean to eliminate DC
signal1 = signal1 - np.mean(signal1)


## create the Gaussian kernel
# full-width half-maximum: the key Gaussian parameter
fwhm = 25 # in ms

# normalized time vector in ms
k = 100
gtime = 1000*np.arange(-k,k)/srate

# create Gaussian window
gauswin = np.exp( -(4*np.log(2)*gtime**2) / fwhm**2 )

# then normalize Gaussian to unit energy
gauswin = gauswin / np.sum(gauswin)


### filter as time-domain convolution

# initialize filtered signal vector
filtsigG = copy.deepcopy(signal1)

# implement the running mean filter
for i in range(k+1,n-k-1):
    # each point is the weighted average of k surrounding points
    filtsigG[i] = np.sum( signal1[i-k:i+k]*gauswin )



## now repeat in the frequency domain

# compute N's
nConv = n + 2*k+1 - 1

# FFTs
dataX = scipy.fftpack.fft(signal1,nConv)
gausX = scipy.fftpack.fft(gauswin,nConv)

# IFFT
convres = np.real( scipy.fftpack.ifft( dataX*gausX ) )

# cut wings
convres = convres[k:-k]

# frequencies vector
hz = np.linspace(0,srate,nConv)

### time-domain plot

# lines
plt.plot(time,signal1,'r',label='Signal')
plt.plot(time,filtsigG,'k*',label='Time-domain')
plt.plot(time,convres,'bo',label='Spectral mult.')
plt.xlabel('Time (s)')
plt.ylabel('amp. (a.u.)')
plt.legend()
plt.show()




### frequency-domain plot

# plot Gaussian kernel
plt.figure()
plt.plot(hz,np.abs(gausX)**2)
plt.xlim([0,30])
plt.ylabel('Gain'), plt.xlabel('Frequency (Hz)')
plt.title('Power spectrum of Gaussian')
plt.show()


# raw and filtered data spectra
plt.figure()
plt.plot(hz,np.abs(dataX)**2,'rs-',label='Signal')
plt.plot(hz,np.abs(dataX*gausX)**2,'bo-',label='Conv. result')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
plt.legend()
plt.title('Frequency domain')
plt.xlim([0,25])
plt.ylim([0,400000])
plt.show()