import numpy as np
import matplotlib.pyplot as plt

srate = 1000
time = np.arange(0,3,1/srate)
noiseamp = 5

ampl   = np.interp(np.linspace(0,15,len(time)),np.arange(0,15),np.random.rand(15)*30) #creating the signal
noise = noiseamp*np.random.randn(len(time)) #creating the noise
signal = ampl + noise  #creating the noisy signal

#the key parameter for Gaussian function
fwhm = 25

k = 40
gtime = 1000*np.arange(-k,k)/srate

gauswin = np.exp( -(4*np.log(2)*gtime**2) / fwhm**2 )

pstPeakHalf = k + np.argmin( (gauswin[k:]-.5)**2 )
prePeakHalf = np.argmin( (gauswin-.5)**2 )

empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]

fig, axs = plt.subplots(2)
fig.suptitle('Gaussian Smoothing')
# show the Gaussian
axs[0].plot(gtime,gauswin,'ko-')
axs[0].plot([gtime[prePeakHalf],gtime[pstPeakHalf]],[gauswin[prePeakHalf],gauswin[pstPeakHalf]],'m')

# then normalize Gaussian to unit energy
gauswin = gauswin / np.sum(gauswin)

# Initializing new filtered signal
filtered_signal = np.copy(signal)
n = len(time)

for i in range(k+1,n-k-1):
    filtered_signal[i] = np.sum(signal[i-k:i+k]*gauswin)

axs[1].plot(time,signal,'r', label='Original')
axs[1].plot(time,filtered_signal, 'b', label='Gaussian-filtered')
axs[1].legend()


