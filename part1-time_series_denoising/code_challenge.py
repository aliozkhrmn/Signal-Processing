import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal
from scipy import *
import copy

data = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\time-series-denoising\\sigprocMXC_TimeSeriesDenoising\\denoising_codeChallenge.mat')
original = data['origSignal'][0] # original signal
cleaned = data['cleanedSignal'][0] # cleaned signal
timevec = np.arange(0,4000)


suprathresh = np.where((original>1) | (original<-1))[0] # eliminating the spikes

k = 25

filtered = copy.deepcopy(original)

for i in range(len(suprathresh)):
    lowbound = max(0, suprathresh[i]-k)
    upbound = min(suprathresh[i]+k, len(original))

    filtered[suprathresh[i]] = np.median(original[lowbound:upbound]) # applying median filter to original signaş


filtered2 = copy.deepcopy(filtered)
k = 85 # best k after some trial

for i in range(len(filtered)): # mean filter considering the edge effects
    lowbnd = max(0, i-k)
    upbnd = min(len(filtered), i+k)
    filtered2[i] = np.mean(filtered[lowbnd:upbnd])



plt.plot(timevec, cleaned, 'r', label='cleaned signal')
plt.plot(timevec, filtered2, 'b', label='filtered signal')
plt.legend()
plt.show()
