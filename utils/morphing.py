import librosa
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.special import expn
from scipy.fftpack import ifft
import numpy as np


def change_pitch(filename, n_steps, sample_rate=16000):
    """
    :param waveform: source wav audio
    :param sample_rate: frequencies per sec
    :param n_steps: steps to move the pitch (can be <0)
    :return: morphed waveform
    """
    waveform, sample_rate = librosa.load(filename, sample_rate)
    y_shifted = librosa.effects.pitch_shift(waveform, sample_rate, n_steps)
    librosa.output.write_wav(f'new_{filename}', y_shifted, sample_rate)
    return y_shifted


def change_volume(filename, n=10):
    """
    :param n: n dB louder
    (instead of "quiter" -- proof https://stackoverflow.com/questions/43679631/python-how-to-change-audio-volume)
    """
    song = AudioSegment.from_wav(filename)

    song = song + n

    # save the output
    song.export(filename, "wav")


def overlap(filename_1, filename_2):
    sound1 = AudioSegment.from_file(filename_1)
    sound2 = AudioSegment.from_file(filename_2)

    combined = sound1.overlay(sound2)
    combined.export(f'new_{filename}', format='wav')


def change_speed(filename, rate):
    # tmp, sr = librosa.load(filename)
    # librosa.output.write_wav('filename', tmp, int(sr * rate))

    sound = AudioSegment.from_file(filename)
    sound = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * rate)})

    sound.export(filename, "wav")


def change_noise_level():
    pass


def change_tremor_level(waveform, sample_rate, level):
    pass


def logMMSE(inputFilePath, outputFilePath):
    [sample_rate, sample_data] = wavfile.read(inputFilePath, True)
    len = np.int(np.floor(20 * sample_rate * 0.001))
    if len % 2 == 1:
        len += 1

    perc = 50
    len1 = np.floor(len * perc * 0.01)
    len2 = len - len1

    win = np.hanning(len)
    win = win * len2 / sum(win)

    nFFT = len << 2
    noise_mean = np.zeros([nFFT, 1])
    dtype = 2 << 14
    j = 0

    for i in range(1, 7):
        s1 = j
        s2 = j + np.int(len)

        batch = sample_data[s1: s2] / dtype

        X = win * batch

        foo = np.fft.fft(X, np.int(nFFT))

        noise_mean += np.abs(foo.reshape(foo.shape[0], 1))

        j += len

    noise_mu = np.square(noise_mean / 6)

    x_old = np.zeros([np.int(len1), 1]);
    Nframes = np.floor(sample_data.shape[0] / len2) - np.floor(len / len2)
    xfinal = np.zeros([np.int(Nframes * len2), 1]);
    k = 0
    aa = 0.98
    mu = 0.98
    eta = 0.15

    ksi_min = 10 ** (-25 * 0.1)

    for n in range(0, np.int(Nframes)):

        s1 = k
        s2 = k + np.int(len)

        batch = sample_data[s1: s2] / dtype
        insign = win * batch

        spec = np.fft.fft(insign, nFFT)

        sig = abs(spec)
        sig2 = sig ** 2

        gammak = np.divide(sig2.reshape(sig2.shape[0], 1), noise_mu.reshape(noise_mu.shape[0], 1))
        gammak[gammak > 40] = 40

        foo = gammak - 1
        foo[foo < 0] = 0

        if 0 == n:
            ksi = aa + (1 - aa) * foo
        else:

            # a priori SNR
            ksi = aa * Xk_prev / noise_mu + (1 - aa) * foo

            # limit ksi to - 25 db
            ksi[ksi < ksi_min] = ksi_min

        log_sigma_k = gammak * ksi / (1 + ksi) - np.log(1 + ksi)
        vad_decision = sum(log_sigma_k) / len

        # noise only frame found
        if vad_decision < eta:
            noise_mu = mu * noise_mu + (1 - mu) * sig2.reshape([sig2.shape[0], 1])

        # == = end of vad == =

        # Log - MMSE estimator
        A = ksi / (1 + ksi)
        vk = A * gammak

        ei_vk = 0.5 * expn(1, vk)
        hw = A * np.exp(ei_vk)

        sig = sig.reshape([sig.shape[0], 1]) * hw
        Xk_prev = sig ** 2

        xi_w = ifft(hw * spec.reshape([spec.shape[0], 1]), nFFT, 0)
        xi_w = np.real(xi_w)

        xfinal[k: k + np.int(len2)] = x_old + xi_w[0: np.int(len1)]
        x_old = xi_w[np.int(len1): np.int(len)]

        k = k + np.int(len2)

    wavfile.write(outputFilePath, sample_rate, xfinal)


def get_params(age, gender='fem'):
    if gender == 'fem':
        if age == 10:
            return {'n_steps': 3.5,
                    'volume_rate': 5,
                    'speed_rate': 1.05}
        elif age == 16:
            return {'n_steps': 1.5,
                    'volume_rate': 5,
                    'speed_rate': 1}
        elif age == 30:
            return {'n_steps': -1.5,
                    'volume_rate': -5,
                    'speed_rate': 0.95}
    else:
        if age == 30:
            return {'n_steps': -0.8,
                    'volume_rate': -5,
                    'speed_rate': 0.95}


def morph(filename, gender, age, sample_rate=16000, params=None):
    waveform, sample_rate = librosa.load(filename, sample_rate)

    params = get_params(age, gender) if params is None else params

    y_shifted = librosa.effects.pitch_shift(waveform, sample_rate, n_steps=float(params['n_steps']))
    librosa.output.write_wav(f'new_{filename}', y_shifted, sample_rate)

    change_volume(f'new_{filename}', n=params['volume_rate'])

    change_speed(f'new_{filename}', rate=params['speed_rate'])
