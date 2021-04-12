import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

n = 2000
signal = np.cumsum(np.random.randn(n)) + np.linspace(-30,30,n) # signal with linear trending

# linear detrending
detsignal = scipy.signal.detrend(signal)

# get means
omean = np.mean(signal) # original mean
dmean = np.mean(detsignal) # detrended mean

# plotting
plt.plot(range(0,n), signal, label='Original, mean=%d' %omean)
plt.plot(range(0,n), detsignal, label='Detrended mean=%d'%dmean)
plt.legend()
plt.show()