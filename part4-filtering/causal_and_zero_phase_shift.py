import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy
import copy

# create a simple signal
data = np.concatenate((np.zeros(100),np.cos(np.linspace(np.pi/2,5*np.pi/2,10)),np.zeros(100)),axis=0)
n = len(data)

plt.figure(0)
# plot it and its power spectrum
plt.subplot(121)
plt.plot(range(0,n),data,'ko-')
plt.xlim([0,n+1])
plt.title('Original signal')
plt.xlabel('Time points (a.u.)')

plt.subplot(122)
plt.plot(np.linspace(0,1,n),np.abs(scipy.fftpack.fft(data)),'ko-')
plt.xlim([0,.5])
plt.xlabel('Frequency (norm.)')
plt.ylabel('Energy')
plt.title('Frequency-domain signal representation')
plt.show()


plt.figure(1)

plt.subplot(141)
fkern = signal.firwin(51,.6)
fdata = signal.lfilter(fkern,1,data)
plt.plot(range(0,n),data,label='Original')
plt.plot(range(0,n),fdata,label='Forward filtered')
plt.legend()
plt.show()

# flip the signal backwards
fdataFlip = fdata[::-1]
# and show its spectrum
plt.subplot(142)
plt.plot(np.linspace(0,1,n),np.abs(scipy.fftpack.fft(data)),'ko-')
plt.plot(np.linspace(0,1,n),np.abs(scipy.fftpack.fft(fdataFlip)),'r')
plt.xlim([0,.5])
plt.show()


# filter the flipped signal
fdataF = signal.lfilter(fkern,1,fdataFlip)
plt.subplot(143)
plt.plot(range(0,n),data,label='Original')
plt.plot(range(0,n),fdataF,label='Backward filtered')
plt.legend()
plt.show()

# finally, flip the double-filtered signal
fdataF = fdataF[::-1]
plt.subplot(144)
plt.plot(range(0,n),data,label='Original')
plt.plot(range(0,n),fdataF,label='Zero-phase filtered')
plt.legend()
plt.show()
