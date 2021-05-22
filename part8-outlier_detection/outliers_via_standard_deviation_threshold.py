import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.io as sio
from scipy.interpolate import griddata
import copy

## signal is log-normal noise
N = 10000
time = np.arange(0,N)/N
signal1 = np.exp( .5*np.random.randn(N) )

# add some random outiers
nOutliers = 50
randpnts = np.random.randint(0,N,nOutliers)
signal1[randpnts] = np.random.rand(nOutliers) * (np.max(signal1)-np.min(signal1))*10

# show the signal
plt.plot(time,signal1,'ks-')

# auto-threshold based on mean and standard deviation
threshold = np.mean(signal1) + 3*np.std(signal1)
plt.plot([time[0],time[-1]],[threshold,threshold],'b--')
plt.show()

#####################################

## interpolate outlier points

# remove supra-threshold points
outliers = signal1 > threshold

# and interpolate missing points
signalR = copy.deepcopy( signal1 )
signalR[outliers] = griddata(time[~outliers], signal1[~outliers], time[outliers], method='cubic')

# and plot the new results
plt.figure()
plt.plot(time,signal1,'k-',label='signal1')
plt.plot(time,signalR,'ro-',label='signalR')
plt.legend()

## optional zoom
#plt.xlim([.1,.2])

plt.show()
