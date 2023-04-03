'''
Developer: Jiaxin Li
E-mail: 1319376761@qq.com
Github: https://github.com/chuqingi/Guided_filtering_dereverberation
Description: Guided spectrogram filtering for single channel speech dereverberation
Reference:
[1] Guided spectrogram filtering for speech dereverberation
'''
import numpy as np
from scipy.io import wavfile
from Guided_spectrogram_filtering import gsf


class Param:
    def __init__(self):
        self.r1 = 1
        self.r2 = 4
        self.beta = 0.8  # (8) β forgetting factor
        self.epsilon = 0.64  # (16) ε
        self.alpha = 1.2  # (20) α
        self.gain_min = 0.1827
        self.gain_max = 1
        self.wlen = 512
        self.overlap = 256
        self.inc = 256


params = Param()
fs, revspeech = wavfile.read('revspeech.wav')
dereverb = gsf(revspeech, params)
wavfile.write('dereverb.wav', fs, dereverb.astype(np.int16))