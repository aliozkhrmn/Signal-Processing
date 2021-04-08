import numpy as np
import matplotlib.pyplot as plt
import copy

# our signal
n = 2000
signal = np.cumsum(np.random.randn(n))

# proportion of time points to replace with noise
propnoise = 0.05

#find noise points
noisepnts = np.random.permutation(n)
noisepnts = noisepnts[0:int(n*propnoise)]

# replace noise in signal
signal[noisepnts] =  50 + np.random.rand(len(noisepnts))*100

# view histogram for threshold
# plt.hist(signal, 100)
# plt.show()

# valeus which are greater than threshold will be considered as spike noise
threshold = 40

# finding data values above threshold
suprathresh = np.where(signal > threshold)[0]

k = 20 # median window size
# initializing the filtered signal
filtsig = copy.deepcopy(signal)

for ti in range(len(suprathresh)):
    lowbnd = np.max((0, suprathresh[ti]-k) )
    uppbnd = np.min((n+1, suprathresh[ti]+k))

    filtsig[suprathresh[ti]] = np.median(signal[lowbnd:uppbnd]) # calculating the median filter

plt.plot(range(0,n), signal, 'r', label='Original signal')
plt.plot(range(0,n), filtsig, 'b', label='Median filtered signal')
plt.legend()
plt.show()