import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.fftpack
from scipy import signal
from scipy.interpolate import griddata
import copy

## Laplace distribution

# parameters
srate  = 100
tv     = np.arange(-5,5,1/srate)
npnts  = len(tv)

# signal components
laplace  = 1-np.exp(-np.abs(tv))
fastsine = .25*np.sin(2*np.pi*tv*15)

# combine into one signal (no noise)
signal1 = laplace + fastsine

# power spectrum (O = original)
hzO = np.linspace(0,srate/2,int(np.floor(npnts/2)+1))
signalO_pow = np.abs(scipy.fftpack.fft(signal1)/npnts)**2
signalO_pow = signalO_pow[:len(hzO)]


# time domain signal
plt.subplot(211)
plt.plot(tv,signal1,'ko-')
plt.title('Time domain')
## optional manual zoom:
#plt.xlim([0,1])

# show power spectrum
plt.subplot(212)
plt.plot(hzO,signalO_pow,'k-')
plt.yscale('log')
plt.title('Frequency domain')
plt.show()


#########################


## downsample by a factor

dnsampleFactor = 4
newSrate = srate/dnsampleFactor

# new time vector after upsampling
newTv = np.arange(-5,5,1/newSrate)
newPnts = len(newTv)



### downsample WITHOUT low-pass filtering (bad idea!!)
signal_dsB = signal1[:-1:dnsampleFactor]

# power spectrum (B = bad)
hz_ds = np.linspace(0,newSrate/2,int(np.floor(newPnts/2)+1))
signal_dsB_pow = np.abs(scipy.fftpack.fft(signal_dsB)/newPnts)**2
signal_dsB_pow = signal_dsB_pow[:len(hz_ds)]


### low-pass filter at new Nyquist frequency! (good idea!!)
fkern = signal.firwin(int(14*newSrate/2),newSrate/2,fs=srate,pass_zero=True)
fsignal = signal.filtfilt(fkern,1,signal1)

# now downsample
signal_dsG = fsignal[:-1:dnsampleFactor]

# power spectrum (G = good)
signal_dsG_pow = np.abs(scipy.fftpack.fft(signal_dsG)/newPnts)**2
signal_dsG_pow = signal_dsG_pow[:len(hz_ds)]

fsignal_pow = np.abs(scipy.fftpack.fft(fsignal)/npnts)**2
fsignal_pow = fsignal_pow[:len(hz_ds)]



# plot in the time domain
plt.figure()
plt.subplot(121)
plt.plot(tv,signal1,'ko-',label='Original')
plt.plot(newTv,.02+signal_dsB,'m^-',label='DS bad')
plt.plot(newTv,.04+signal_dsG,'gs-',label='DS good')
plt.legend()
## optional change in xlimit to zoom in
#plt.xlim([1,2])
plt.title('Time domain')
plt.show()

# plot in the frequency domain
plt.subplot(122)
plt.plot(hzO,signalO_pow,'ko-',label='Original')
plt.plot(hz_ds,signal_dsB_pow,'m^-',label='DS bad')
plt.plot(hz_ds,signal_dsG_pow,'gs-',label='DS good')
plt.legend()
plt.title('Frequency domain')
plt.yscale('log')
plt.show()

##########################

## using Python's resample function

# use resample function
signal_dsP = signal.resample(signal1,newPnts)


# power spectrum (P=Python)
signal_dsP_pow = np.abs(scipy.fftpack.fft(signal_dsP)/newPnts)**2
signal_dsP_pow = signal_dsP_pow[:len(hz_ds)]


# plot in the time domain
plt.figure()
plt.subplot(121)
plt.plot(tv,signal1,'ko-',label='Original')
plt.plot(newTv,.02+signal_dsB,'m^-',label='DS bad')
plt.plot(newTv,.04+signal_dsG,'gs-',label='DS good')
plt.plot(newTv,.06+signal_dsP,'b-',label='DS Pyth')
plt.legend()
## optional change in xlimit to zoom in
plt.xlim([1,2])
plt.title('Time domain')
plt.show()


# frequency domain
plt.subplot(122)
plt.plot(hzO,signalO_pow,'ko-',label='Original')
plt.plot(hz_ds,signal_dsB_pow,'m^-',label='DS bad')
plt.plot(hz_ds,signal_dsG_pow,'gs-',label='DS good')
plt.plot(hz_ds,signal_dsP_pow,'bs-',label='DS Pyth')
plt.legend()
plt.title('Frequency domain')
plt.yscale('log')
plt.show()
