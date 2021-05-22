import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy

# simulation parameters
srate    = 1324    # Hz
peakfreq =    7    # Hz
fwhm     =    5    # Hz
npnts    = srate*2 # time points
timevec  = np.arange(0,npnts)/srate # seconds

# frequencies
hz = np.linspace(0,srate,npnts)
s  = fwhm*(2*np.pi-1)/(4*np.pi) # normalized width
x  = hz-peakfreq                # shifted frequencies
fg = np.exp(-.5*(x/s)**2)       # gaussian


# Fourier coefficients of random spectrum
fc = np.random.rand(npnts) * np.exp(1j*2*np.pi*np.random.rand(npnts))

# taper with Gaussian
fc = fc * fg

# go back to time domain to get signal
signal1 = 2*np.real( scipy.fftpack.ifft(fc) )*npnts


### plot
plt.plot(timevec,signal1,'k')
plt.xlabel('Time (s)')
plt.show()


####################################

## now randomly sample from this "continuous" time series

# initialize to empty
sampSig = []

# random sampling intervals
sampintervals = np.append(1, np.cumsum(np.ceil(np.exp(4 * np.random.rand(npnts)))))
sampintervals = sampintervals[sampintervals < np.array(npnts)]  # remove points beyond the data

# loop through sampling points and "measure" data
for i in range(0, len(sampintervals)):
    # "real world" measurement
    nextdat = signal1[int(sampintervals[i])], timevec[int(sampintervals[i])]

    # put in data matrix
    sampSig.append(nextdat)

# needs to be numpy array
sampSig = np.array(sampSig)

## upsample to original sampling rate
# interpolate using griddata
newsignal = griddata(sampSig[:, 1], sampSig[:, 0], timevec, method='cubic')

### and plot everything
plt.figure()
plt.plot(timevec, signal1, 'k', label='"Analog"')
plt.plot(sampSig[:, 1], sampSig[:, 0], 'ro', label='Measured')
plt.plot(timevec, newsignal, 'm.', label='Upsampled')
plt.legend()

## optional zoom
# plt.xlim([1,1.1])

plt.show()
