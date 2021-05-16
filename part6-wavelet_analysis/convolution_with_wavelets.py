import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy

## general simulation parameters

fs = 1024
npnts = fs*5 # 5 seconds

# centered time vector
timevec = np.arange(0,npnts)/fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))

### create wavelets

# parameters
freq = 4 # peak frequency
csw  = np.cos(2*np.pi*freq*timevec) # cosine wave
fwhm = .5 # full-width at half-maximum in seconds
gaussian = np.exp( -(4*np.log(2)*timevec**2) / fwhm**2 ) # Gaussian


## Morlet wavelet
MorletWavelet = csw * gaussian


## Haar wavelet
HaarWavelet = np.zeros(npnts)
HaarWavelet[np.argmin(timevec**2) : np.argmin( (timevec-.5)**2 )] = 1
HaarWavelet[np.argmin((timevec-.5)**2) : np.argmin( (timevec-1-1/fs)**2 )] = -1


## Mexican hat wavelet
s = .4
MexicanWavelet = (2/(np.sqrt(3*s)*np.pi**.25)) * (1- (timevec**2)/(s**2) ) * np.exp( (-timevec**2)/(2*s**2) )


## convolve with random signal

# signal
signal1 = scipy.signal.detrend(np.cumsum(np.random.randn(npnts)))

# convolve signal with different wavelets
morewav = np.convolve(signal1,MorletWavelet,'same')
haarwav = np.convolve(signal1,HaarWavelet,'same')
mexiwav = np.convolve(signal1,MexicanWavelet,'same')

# amplitude spectra
morewaveAmp = np.abs(scipy.fftpack.fft(morewav)/npnts)
haarwaveAmp = np.abs(scipy.fftpack.fft(haarwav)/npnts)
mexiwaveAmp = np.abs(scipy.fftpack.fft(mexiwav)/npnts)



### plotting
# the signal
plt.plot(timevec,signal1,'k')
plt.title('Signal')
plt.xlabel('Time (s)')
plt.show()


# the convolved signals
plt.figure()
plt.subplot(211)
plt.plot(timevec,morewav,label='Morlet')
plt.plot(timevec,haarwav,label='Haar')
plt.plot(timevec,mexiwav,label='Mexican')
plt.title('Time domain')
plt.legend()


# spectra of convolved signals
plt.subplot(212)
plt.plot(hz,morewaveAmp[:len(hz)],label='Morlet')
plt.plot(hz,haarwaveAmp[:len(hz)],label='Haar')
plt.plot(hz,mexiwaveAmp[:len(hz)],label='Mexican')
plt.yscale('log')
plt.xlim([0,40])
plt.legend()
plt.xlabel('Frequency (Hz.)')
plt.show()
