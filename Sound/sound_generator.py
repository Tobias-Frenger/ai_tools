# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:09:01 2020

@author: Tobias
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sampleRate = 22050
frequency = 100
duration = 10

t1 = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
y = np.sin(frequency * 2 * np.pi * t1)
y = y.astype("float32")

sampleRate = 22050
frequency1 = frequency + 0.2
t2 = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
y2 = np.sin(frequency1 * 2 * np.pi * t2)
y2 = y2.astype("float32")

sampleRate = 22050
frequency2 = frequency - 3
t3 = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
y3 = np.sin(frequency2 * 2 * np.pi * t3)
y3 = y3.astype("float32")

sampleRate = 22050
frequency3 = frequency - 0.73
t4 = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
y4 = np.sin(frequency3 * 2 * np.pi * t4)
y4 = y4.astype("float32")

sampleRate = 22050
frequency4 = frequency + 7
t5 = np.linspace(0, duration, sampleRate * duration)  #  Produces a 5 second Audio-File
y5 = np.sin(frequency4 * 2 * np.pi * t5)
y5 = y5.astype("float32")

res = y + y2
res = res/2
res = res + y3
res = res / 2
res = res + y4
res = res / 2
res = res + y5
res = res / 2

print(y.size)
wavfile.write('Sine.wav', sampleRate, res)

plt.plot(res)
plt.xlabel('Time(T)')
plt.ylabel('Amplitude(A)')
plt.show()


#y = np.append(y, y2)
#y = np.append(y, [[y2],[y3]])

