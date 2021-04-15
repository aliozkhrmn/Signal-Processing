import numpy as np
import scipy.io as sio
import scipy.signal
from scipy import *
import copy
import matplotlib.pyplot as plt

matdat = sio.loadmat('C:\\Users\\ali\\Desktop\\signal_processing_python\\time-series-denoising\\sigprocMXC_TimeSeriesDenoising\\templateProjection.mat')
EEGdat = matdat['EEGdat']
eyedat = matdat['eyedat']
timevec = matdat['timevec'][0]

MN = np.shape(EEGdat)  # matrix sizes

# initialize residual data
resdat = np.zeros(np.shape(EEGdat))

# loop over trials
for triali in range(0, MN[1]):
    # build the least-squares model as intercept and EOG from this trial
    X = np.column_stack((np.ones(MN[0]), eyedat[:, triali]))

    # compute regression coefficients for EEG channel
    b = np.linalg.solve(np.matrix.transpose(X) @ X, np.matrix.transpose(X) @ EEGdat[:, triali])

    # predicted data
    yHat = X @ b

    # new data are the residuals after projecting out the best EKG fit
    resdat[:, triali] = EEGdat[:, triali] - yHat

# trial averages
# plt.plot(timevec,np.mean(eyedat,axis=1),label='EOG')
# plt.plot(timevec,np.mean(EEGdat,axis=1),label='EEG')
# plt.plot(timevec,np.mean(resdat,1),label='Residual')
#
# plt.xlabel('Time (ms)')
# plt.legend()
# plt.show()

# show all trials in a map
clim = [-1,1]*20

plt.subplot(131)
plt.imshow(eyedat.T)
plt.title('EOG')

plt.subplot(132)
plt.imshow(EEGdat.T)
plt.title('EEG')

plt.subplot(133)
plt.imshow(resdat.T)
plt.title('Residual')
plt.show()