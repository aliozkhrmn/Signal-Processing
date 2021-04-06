import numpy as np
import matplotlib.pyplot as plt

srate = 1000
time = np.arange(0,3,1/srate)
noiseamp = 5

ampl   = np.interp(np.linspace(0,15,len(time)),np.arange(0,15),np.random.rand(15)*30) #creating the signal
noise = noiseamp*np.random.randn(len(time)) #creating the noise
signal = ampl + noise  #creating the noisy signal

# Initializing new filtered signal
filtered_signal = np.copy(signal)

k = 20 #total window size is: 2*k+1 = 41
n = len(time)

# Applying mean filter:
for i in range(k, n-k-1):
    filtered_signal[i] = np.mean(signal[i-k:i+k])

plt.plot(time, signal, label='Original Signal')
plt.plot(time, filtered_signal, label='Filtered Signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Mean Filter')
plt.show()