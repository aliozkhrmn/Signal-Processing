import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import scipy
import copy

# filter parameters
srate   = 1024 # hz
nyquist = srate/2
frange  = [20,45]
transw  = .1
order   = int( 5*srate/frange[0] )

# force odd order
if order%2==0:
    order += 1

### --- NOTE: Python's firwin corresponds to MATLAB's fir1 --- ###

# filter kernel
filtkern = signal.firwin(order,frange,fs=srate,pass_zero=False)

plt.figure(0)
plt.subplot(1,3,1)
# time-domain filter kernel
plt.plot(filtkern)
plt.xlabel('Time points')
plt.title('Filter kernel (firwin)')
#plt.show()




# compute the power spectrum of the filter kernel
filtpow = np.abs(scipy.fftpack.fft(filtkern))**2
# compute the frequencies vector and remove negative frequencies
hz      = np.linspace(0,srate/2,int(np.floor(len(filtkern)/2)+1))
filtpow = filtpow[0:len(hz)]



# plot amplitude spectrum of the filter kernel
plt.subplot(1,3,2)
plt.plot(hz,filtpow,'ks-',label='Actual')
plt.plot([0,frange[0],frange[0],frange[1],frange[1],nyquist],[0,0,1,1,0,0],'ro-',label='Ideal')
plt.xlim([0,frange[0]*4])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency response of filter (firwin)')
#plt.show()



# Same as above but logarithmically scaled
plt.subplot(1,3,3)
plt.plot(hz,10*np.log10(filtpow),'ks-',label='Actual')
plt.plot([frange[0],frange[0]],[-100,5],'ro-',label='Ideal')
plt.xlim([0,frange[0]*4])
plt.ylim([-80,5])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency response of filter (firwin)')
plt.show()

plt.figure(1)
## effects of the filter kernel order

# range of orders
orders = np.round(np.linspace((srate / frange[0]) / (srate / 1000), (15 * srate / frange[0]) / (srate / 1000), 10))

# initialize
fkernX = np.zeros((len(orders), 1000))
hz = np.linspace(0, srate, 1000)

for oi in range(0, len(orders)):
    # make sure order is odd-length
    ord2use = orders[oi] + (1 - orders[oi] % 2)

    # create filter kernel
    fkern = signal.firwin(int(ord2use), frange, fs=srate, pass_zero=False)

    # take its FFT
    fkernX[oi, :] = np.abs(scipy.fftpack.fft(fkern, 1000)) ** 2

    # show in plot
    time = np.arange(0, ord2use) / srate
    time = time - np.mean(time)
    plt.subplot(1,3,1)
    plt.plot(time, fkern + .01 * oi)

plt.xlabel('Time (ms)')
plt.title('Filter kernels with different orders')
#plt.show()

plt.subplot(1,3,2)
plt.plot(hz, fkernX.T)
plt.plot([0, frange[0], frange[0], frange[1], frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'k')
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation')
plt.title('Frequency response of filter (firwin)')
#plt.show()

plt.subplot(1,3,3)
plt.plot(hz, 10 * np.log10(fkernX.T))
plt.xlim([0, 100])
plt.title('Same as above but logscale')
plt.show()