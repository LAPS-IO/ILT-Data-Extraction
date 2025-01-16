import math
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt


# Generating a sample musical note signal
FS = 1 / 1800
KS = 10

# Open time series
df = pd.read_csv('/home/caio_mb_88/Downloads/dftest.csv')
signal = list(df['chaetoceros'])
duration = len(signal) / FS
N = math.ceil(FS * duration)

# Applying FFT
rfft_result = scipy.fft.rfft(signal)
hz_freq = scipy.fft.rfftfreq(N, d=1 / FS)
uhz_freq = 1000000 * hz_freq

kernel = np.ones(KS) / KS
moving_avg = np.convolve(np.abs(rfft_result), kernel, 'same')
moving_avg = np.convolve(moving_avg, kernel, 'same')
moving_avg = np.convolve(moving_avg, kernel, 'same')

# Plotting the spectrum
#plt.plot(uhz_freq, np.abs(rfft_result))
plt.plot(uhz_freq, np.abs(rfft_result))
plt.title('FFT of chaetoceros')
plt.xlabel('Frequency (uHz)')
plt.ylabel('Amplitude')
plt.show()
