import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt

# data downloaded from https://trends.google.com/trends/explore?date=today%205-y&geo=US&q=signal%20processing
searchdata = [69,77,87,86,87,71,70,92,83,73,76,78,56,75,68,60,30,44,58,69,82,76,73,60,71,86,72,55,56,65,73,71,71,71,62,65,57,54,54,60,49,59,58,46,50,62,60,65,67,60,70,89,78,94,86,80,81,73,100,95,78,75,64,80,53,81,73,66,26,44,70,85,81,91,85,79,77,80,68,67,51,78,85,76,72,87,65,59,60,64,56,52,71,77,53,53,49,57,61,42,58,65,67,93,88,83,89,60,79,72,79,69,78,85,72,85,51,73,73,52,41,27,44,68,77,71,49,63,72,73,60,68,63,55,50,56,58,74,51,62,52,47,46,38,45,48,44,46,46,51,38,44,39,47,42,55,52,68,56,59,69,61,51,61,65,61,47,59,47,55,57,48,43,35,41,55,50,76,56,60,59,62,56,58,60,58,61,69,65,52,55,64,42,42,54,46,47,52,54,44,31,51,46,42,40,51,60,53,64,58,63,52,53,51,56,65,65,61,61,62,44,51,54,51,42,34,42,33,55,67,57,62,55,52,48,50,48,49,52,53,54,55,48,51,57,46,45,41,55,44,34,40,38,41,31,41,41,40,53,35,31]
N = len(searchdata)

# possible normalizations...
searchdata = searchdata - np.mean(searchdata)

# power
searchpow = np.abs( scipy.fftpack.fft( searchdata )/N )**2
hz = np.linspace(0,52,N)

# plt.plot(searchdata,'ko-')
# plt.xlabel('Time (weeks)')
# plt.ylabel('Search volume')
# plt.show()

plt.plot(hz,searchpow,'ms-')
plt.xlabel('Frequency (norm.)')
plt.ylabel('Search power')
plt.xlim([0,12])
plt.show()