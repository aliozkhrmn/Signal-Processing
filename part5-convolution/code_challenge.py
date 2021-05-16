import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.fftpack

srate = 1000
time = np.arange(0,3,1/srate)
noiseamp = 5

#creating the signal
ampl   = np.interp(np.linspace(0,15,len(time)),np.arange(0,15),np.random.rand(15)*30) #creating the signal
noise = noiseamp*np.random.randn(len(time)) #creating the noise
signal = ampl + noise  #creating the noisy signal
n = len(time)

# subtract mean to eliminate DC
signal1 = signal - np.mean(signal)

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


# using time domain denoising technique with mean filter
k = 20
# Initializing new filtered signal
filtered_signal = np.copy(signal)

for i in range(k, n-k-1):
    filtered_signal[i] = np.mean(signal1[i - k:i + k])



# plotting
plt.plot(time,signal1,'r',label='Signal')
plt.plot(time,filtered_signal,'k*',label='Time-domain')
plt.plot(time,convres,'bo',label='Spectral mult.')
plt.xlabel('Time (s)')
plt.ylabel('amp. (a.u.)')
plt.legend()
plt.show()
