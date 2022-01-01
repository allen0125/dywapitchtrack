import os
from ctypes import *
import librosa
import pylab as pl
import numpy as np
import scipy.signal as signal


class DyWaPitchTracker(Structure):
    _fields = [
        ("_prevPitch", c_double),
        ("_pitchConfidence", c_int),
    ]

pitchtracker = DyWaPitchTracker()
ptr_pitchtracker = pointer(pitchtracker)

libpitch = cdll.LoadLibrary('libpitch.so')
libpitch.dywapitch_inittracking(ptr_pitchtracker)
libpitch.dywapitch_computepitch.restype = c_double

print(libpitch)

def pitch_detect(audio_path):
    samples, sr = librosa.load(audio_path, sr=44100, mono=True)
    print(sr)
    samples = list(samples)
    len_s = len(samples)

    c_d_samples = c_double * len_s
    samples = c_d_samples(*samples)

    ptr_s = pointer(samples)

    print(len_s)
    cur = 0
    step = 2048

    i = 1
    pitchs = []
    while cur < len_s - step:
        thepitch = libpitch.dywapitch_computepitch(ptr_pitchtracker, ptr_s, cur, step)
        pitchs.append(thepitch)
        cur += 2048 - 336
        i += 1
    
    # return pitchs
    return signal.medfilt(pitchs, 5)


def show(pitchs_1, pitchs_2):
    pl.subplot(311)
    pl.plot([i for i in range(len(pitchs_1))], pitchs_1, "g")
    pl.xlabel("index of pitch")
    pl.ylabel("BGM Pitchs")

    pl.subplot(312)
    pl.plot([i for i in range(len(pitchs_2))], pitchs_2, "g")
    pl.xlabel("index of pitch")
    pl.ylabel("Vocal Pitchs")

    pl.show()


def mape_vectorized_v2(a, b):
    a = np.array(a)
    b = np.array(b)
    mask = a != 0
    return (np.fabs(a - b) / a)[mask].mean()


gretter = '/Users/allen/Desktop/score_sample/id_422/D57CD19978973975AE7E33A01418F852.wav'

# audios = [
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/3e2b54e2d70b46d2b8dab2319b9953f4.m4a',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/9ce0ae91cd6c453f8cebe66ffc4ff062.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/c8843bd6644041d6b8ecfc1eb383c5de.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/ce258c96357c4c6e9a438a404a690610.m4a',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/e32cd25758164d31927be541169dce1e.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/e03392618fc940b599b9c372383a7ce3.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/47c38eb2aa6149a9b86440953e2ec56d.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/20304e9a3f3b49148de78118cb41a8bb.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/a5069266d0a847aeb3c2501e0840980c.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/b899c397d2fe4db9a6098a11bba36868.mp4',
#     '/Users/allen/Project/YaShi/lab/pitch/data/jgzy_378/c93605404d9f4cc292cca9bbb823aa4a.m4a',
# ]
base_dir = '/Users/allen/Desktop/score_sample/id_422/'

def findAllFile(base):
    for _, _, fs in os.walk(base):
        for f in fs:
            yield f

for i in findAllFile(base_dir):
    print(os.path.join(base_dir, i))


bgm_pitch = signal.medfilt(pitch_detect(gretter), 5)
for i in findAllFile(base_dir):
    if i == '.DS_Store': continue
    vocal_dir = os.path.join(base_dir, i)
    if vocal_dir == gretter:
        continue

    audio_pitch = signal.medfilt(pitch_detect(vocal_dir), 5)
    # audio_pitch = [i + 8*12 for i in audio_pitch]
    # show(bgm_pitch, audio_pitch)
    if len(bgm_pitch) > len(audio_pitch):
        bgm_pitch = bgm_pitch[:len(audio_pitch)]
    elif len(bgm_pitch) < len(audio_pitch):
        audio_pitch = audio_pitch[:len(bgm_pitch)]

    print(vocal_dir)
    score = '%0.2f' % ((1 - mape_vectorized_v2(bgm_pitch, audio_pitch)) * 100)
    with open('score.csv', 'a') as f:
        f.write(i + ',' + score + '\n')
