import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy
import copy

# create a signal
N  = 500
hz = np.linspace(0,1,N)
gx = np.exp( -(4*np.log(2)*(hz-.1)/.1)**2 )*N/2
data = np.real(scipy.fftpack.ifft( gx*np.exp(1j*np.random.rand(N)*2*np.pi) ))
data = data + np.random.randn(N)

# plot it and its power spectrum
plt.figure(0)
plt.subplot(121)
plt.plot(range(0,N),data,'k')
plt.title('Original signal')
plt.xlabel('Time (a.u.)')
plt.show()

plt.subplot(122)
plt.plot(hz,np.abs(scipy.fftpack.fft(data))**2,'k')
plt.xlim([0,.5])
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()


plt.figure(1)
## apply a low-pass causal filter

# generate filter kernel
order = 151
fkern = signal.firwin(order,.6)

# zero-phase-shift filter
fdata = signal.lfilter(fkern,1,data) # forward
fdata = signal.lfilter(fkern,1,np.flip(fdata,0)) # reverse
fdata = np.flip(fdata,0) # flip forward


# plot the original signal and filtered version
plt.subplot(121)
plt.plot(range(0,N),data,'k',label='Original')
plt.plot(range(0,N),fdata,'m',label='Filtered, no reflection')
plt.title('Time domain')
plt.legend()

# power spectra
plt.subplot(122)
plt.plot(hz,np.abs(scipy.fftpack.fft(data))**2,'k',label='Original')
plt.plot(hz,np.abs(scipy.fftpack.fft(fdata))**2,'m',label='Filtered, no reflection')
plt.title('Frequency domain')
plt.xlim([0,.5])
plt.legend()
plt.show()


plt.figure(2)

## now with reflection by filter order

# reflect the signal
# data = np.concatenate((np.zeros(100),np.cos(np.linspace(np.pi/2,5*np.pi/2,10)),np.zeros(100)),axis=0)
reflectdata = np.concatenate( (data[order:0:-1],data,data[-1:-1-order:-1]) ,axis=0)

# zero-phase-shift filter on the reflected signal
reflectdata = signal.lfilter(fkern,1,reflectdata)
reflectdata = signal.lfilter(fkern,1,reflectdata[::-1])
reflectdata = reflectdata[::-1]

# now chop off the reflected parts
fdata = reflectdata[order:-order]

# try again with filtfilt
fdata1 = signal.filtfilt(fkern,1,data)

# and plot
plt.subplot(121)
plt.plot(range(0,N),data,'k',label='original')
plt.plot(range(0,N),fdata,'m',label='filtered')
# plt.plot(range(0,N),fdata1,'b',label='filtered1')
plt.xlabel('Time (a.u.)')
plt.title('Time domain')
plt.legend()
plt.show()


# spectra
plt.subplot(122)
plt.plot(hz,np.abs(scipy.fftpack.fft(data))**2,'k',label='Original')
plt.plot(hz,np.abs(scipy.fftpack.fft(fdata))**2,'m',label='Filtered')
plt.legend()
plt.xlim([0,.5])
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency domain')
plt.show()