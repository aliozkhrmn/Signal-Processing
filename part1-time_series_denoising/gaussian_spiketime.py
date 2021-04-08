import numpy as np
import matplotlib.pyplot as plt

# number of spikes in a signal
n = 300

isi = np.round(np.exp( np.random.randn(n))*10)

spikets = np.zeros(int(sum(isi)))

for i in range(0,n):
    spikets[ int(np.sum(isi[0:i])) ] = 1

# plt.plot(spikets)
# plt.xlabel('Time ')
# plt.show()

# Gaussian

# full width at half maximum parameter
fwhm = 25

k=100
gtime = np.arange(-k,k)

# Gaussian window
gauswin = np.exp( -(4*np.log(2)*gtime**2) / fwhm**2 )
gauswin = gauswin / np.sum(gauswin)

filtered_signal = np.zeros(len(spikets))
n = len(spikets)

for i in range(k+1,n-k-1):
    filtered_signal[i] = np.sum(spikets[i-k:i+k]*gauswin)

plt.plot(spikets,'b',label='spikes')
plt.plot(filtered_signal,'r',label='spike p.d.')
plt.legend()
plt.title('Spikes and spike probability density')
plt.show()