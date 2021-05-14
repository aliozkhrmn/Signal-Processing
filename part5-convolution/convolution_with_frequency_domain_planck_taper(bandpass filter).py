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

## create the signal

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

plt.plot(time,signal1)
plt.show()


## create Planck spectral shape

# frequencies
hz = np.linspace(0,srate,n)

# edge decay, must be between 0 and .5
eta = .15

# spectral parameters
fwhm  = 13
peakf = 20

# convert fwhm to indices
mp = np.round( 2*fwhm*n/srate ) # in MATLAB this is np, but np=numpy
pt = np.arange(1,mp+1)

# find center point index
fidx = np.argmin( (hz-peakf)**2 )


# define left and right exponentials
Zl = eta*(mp-1) * ( 1/pt + 1/(pt-eta*(mp-1)) )
Zr = eta*(mp-1) * ( 1/(mp-1-pt) + 1/( (1-eta)*(mp-1)-pt ) )

# create the taper
offset = mp%2
bounds = [ np.floor(eta*(mp-1))-offset , np.ceil((1-eta)*(mp-(1-offset))) ]
plancktaper = np.concatenate( (1/(np.exp(Zl[range(0,int(bounds[0]))])+1) ,np.ones(int(np.diff(bounds)+1)), 1/(np.exp(Zr[range(int(bounds[1]),len(Zr)-1)])+1)) ,axis=0)

# put the taper inside zeros
px = np.zeros( len(hz) )
pidx = range( int(np.max((0,fidx-np.floor(mp/2)+1))) , int(fidx+np.floor(mp/2)-mp%2+1) )
px[np.round(pidx)] = plancktaper


## now for convolution

# FFTs
dataX = scipy.fftpack.fft(signal1)

# IFFT
convres = 2*np.real( scipy.fftpack.ifft( dataX*px ))

# frequencies vector
hz = np.linspace(0,srate,n)

### time-domain plots

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
plt.plot(hz,px,'k')
plt.xlim([0,peakf*2])
plt.ylabel('Gain')
plt.title('Frequency-domain Planck taper')
plt.show()

# raw and filtered data spectra
plt.figure()
plt.plot(hz,np.abs(dataX)**2,'rs-',label='Signal')
plt.plot(hz,np.abs(dataX*px)**2,'bo-',label='Conv. result')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (a.u.)')
plt.legend()
plt.title('Frequency domain')
plt.xlim([0,peakf*2])
plt.ylim([0,1e6])
plt.show()