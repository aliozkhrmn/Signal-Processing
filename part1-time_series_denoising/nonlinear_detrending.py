import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import copy
from scipy import *

n = 10000
t = range(n)
k = 10 # number of poles for random amplitudes

slowdrift = np.interp(np.linspace(1,k,n),np.arange(0,k),100*np.random.randn(k))
signal = slowdrift + 20*np.random.randn(n) # signal with slow drift

## Bayes information criterion to find optimal order

# possible orders
orders = range(5, 40)

# sum of squared errors (sse is reserved!)
sse1 = np.zeros(len(orders))

# loop through orders
for ri in range(len(orders)):
    # compute polynomial (fitting time series)
    yHat = np.polyval(polyfit(t, signal, orders[ri]), t)

    # compute fit of model to data (sum of squared errors)
    sse1[ri] = np.sum((yHat - signal) ** 2) / n

# Bayes information criterion
bic = n * np.log(sse1) + orders * np.log(n)
# best parameter has lowest BIC
bestP = min(bic)
idx = np.argmin(bic)

# plot the BIC
# plt.plot(orders, bic, 'ks-')
# plt.plot(orders[idx], bestP, 'ro')
# plt.xlabel('Polynomial order')
# plt.ylabel('Bayes information criterion')
# plt.show()
# polynomial fit
polycoefs = polyfit(t,signal,orders[idx])

# estimated data based on the coefficients
yHat = polyval(polycoefs,t)

# filtered signal is residual
filtsig = signal - yHat


## plotting
plt.plot(t,signal,'b',label='Original')
plt.plot(t,yHat,'r',label='Polynomial fit')
plt.plot(t,filtsig,'k',label='Filtered')

plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()