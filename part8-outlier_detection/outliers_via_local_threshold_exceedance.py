import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import scipy.io as sio
from scipy.interpolate import griddata
import copy

# data downloaded from:
# http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2017

# import data, etc.
matdat = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\outlier_detection\\forex.mat')
forex  = np.squeeze(matdat['forex'])

N = len(forex)
time = np.arange(0,N)/N


# plot it
plt.plot(time,forex)
plt.xlabel('Time (year)')
plt.ylabel('EUR/USD')

# add global thresholds
threshup = np.mean(forex)+3*np.std(forex)
threshdn = np.mean(forex)-3*np.std(forex)
plt.plot(range(0,N),forex,'b',label='EUR/USD')
plt.plot([0,N],[threshup,threshup],'r--',label='M+3std')
plt.plot([0,N],[threshdn,threshdn],'k--',label='M-3std')
plt.legend()
plt.show()

##########################################


## local threshold

# window size as percent of total signal length
pct_win = 5  # in percent, not proportion!

# convert to indices
k = int(len(forex) * (pct_win / 2 / 100))

# # initialize statistics time series to be the global stats
mean_ts = np.ones(len(time)) * np.mean(forex)
std3_ts = np.ones(len(time)) * np.std(forex)


# # loop over time points
for i in range(0,N):

#     # boundaries
    lo_bnd = np.max((0,i-k))
    hi_bnd = np.min((i+k,N))

#     # compute local mean and std
    mean_ts[i] =  np.mean( forex[range(lo_bnd,hi_bnd)] )
    std3_ts[i] = 3*np.std( forex[range(lo_bnd,hi_bnd)] )


## compute local outliers
outliers = (forex > mean_ts + std3_ts) | (forex < mean_ts - std3_ts)
plt.figure()
# plotting...
plt.plot(time, forex, 'k', label='EUR/USD')
plt.plot(time, mean_ts + std3_ts, 'm--', label='Mean +/- 3std')
plt.plot(time, mean_ts - std3_ts, 'm--')

# and plot those
plt.plot(time[outliers], forex[outliers], 'ro', label='Outliers')

plt.legend()
plt.xlabel('Time (year)')
plt.ylabel('EUR/USD')
plt.title('Using a %d%% window size' % pct_win)
plt.show()