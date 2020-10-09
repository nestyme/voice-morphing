import librosa
from pydub import AudioSegment


def change_pitch(waveform, sample_rate, n_steps):
    """
    :param waveform: source wav audio
    :param sample_rate: frequencies per sec
    :param n_steps: steps to move the pitch (can be <0)
    :return: morphed waveform
    """
    y_shifted = librosa.effects.pitch_shift(waveform, sample_rate, n_steps)
    return y_shifted

def change_volume(filename, n=10):
    """
    :param n: n dB quieter
    """
    song = AudioSegment.from_wav(filename)

    song = song - n

    # save the output
    song.export(filename, "wav")

def change_noize_level():
    pass
def change_tremor_level(waveform, sample_rate, level):
    pass