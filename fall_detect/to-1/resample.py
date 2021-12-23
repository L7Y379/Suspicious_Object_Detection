import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy
#import scikits.samplerate as sk_samplerate
import scipy.signal as scipy_signal
import resampy

P = 48000
Q = 44100

offset = 2000
instfreq = np.exp(np.linspace(np.log(offset+100), np.log(offset+23900), 96000))-offset
deltaphase = 2*np.pi*instfreq/48000
cphase = np.cumsum(deltaphase)
sig = np.sin(cphase)
print(sig.shape)


hhh=scipy_signal.resample(sig, int(len(sig)*Q/P))


print(hhh.shape)
plt.figure(figsize=(15,3))
plt.specgram(scipy_signal.resample(sig, int(len(sig)*Q/P))*30, scale='dB', Fs=44100, NFFT=256)
plt.colorbar()
_=plt.axis((0,2,0,22500))

#data1 = resampy.resample(data1.reshape(-1), data1.shape[0], 300)