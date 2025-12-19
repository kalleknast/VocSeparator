import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


PATH = "/home/hjalmar/Python/call_class/MarmosetVocalizations/"

FNAMES = ["phee_2/typePHEE2_d141207_t094950_087.wav",
          "phee_2/typePHEE2_d141210_t121636_091.wav",
          "phee_3/typePHEE3_d141206_t133129_120.wav",
          "phee_4/typePHEE4_d141206_t142830_148.wav"]
FNAMES = FNAMES[1:]

SCALES = [1, 1, 1, 1]
COLORS = ['darkorange', 'seagreen', 'coral', 'cornflowerblue']

N = len(FNAMES)

plt.ion()


max_len = 380000

fig1 = plt.figure(figsize=(5, 10))
fig2 = plt.figure(figsize=(5, 10))

acc = np.zeros(max_len)

for i, fname in enumerate(FNAMES):

    sr, call = wavfile.read(PATH + fname)
    call = SCALES[i] *  (call.astype(float) / call.max())
    call -= call.mean()

    n = call.shape[0]
    i0 = (max_len - n) // 2
    data = np.zeros(max_len)
    data[i0: i0 + n] = call 
    acc += data 

    ax = fig1.add_subplot(N, 1, i + 1)
    ax.plot(data, c=COLORS[i])
    ax.axis("off")


    ax = fig2.add_subplot(N, 1, i + 1)
    ax.plot(acc, c=[1 - (i+1)/N]*3)
    ax.axis("off")

fig1.savefig('voc_demuxer_individual.svg')
fig2.savefig('voc_demuxer_accumulated.svg')
