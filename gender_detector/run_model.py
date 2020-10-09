import torch
from gender_detector.model import Model
import librosa
import numpy as np
import torch as tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} mode')
model = Model()
model.load_state_dict(
    torch.load(
        "gender_detector/checkpoint/best.pt", map_location={'cuda:0': 'cpu'}))
if device == torch.device('cuda'):
    model.cuda()
else:
    model.cpu()
model.eval()


def spec_to_image(spec, eps=1e-6):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled

def preprocess_sample_inference(amplitudes, sr=16000, max_length=150, device='cpu'):
    spectrogram = librosa.feature.melspectrogram(amplitudes, sr=sr, n_mels=128, fmin=1, fmax=8192)[:, :max_length]
    spectrogram = np.pad(spectrogram, [[0, 0], [0, max(0, max_length - spectrogram.shape[1])]], mode='constant')
    spectrogram = np.array([spec_to_image(np.float32(spectrogram))]).transpose([0, 2, 1])

    return torch.tensor(spectrogram, dtype=torch.float).to(device, non_blocking=True)
def predict(wavfile):
    waveform, _ = librosa.load(wavfile, sr=16000)

    input = preprocess_sample_inference(waveform)
    with torch.no_grad():
        out = model(input).cpu().detach().numpy()
    out = 'female' if out < 0.5 else 'male'
    return out


if __name__ == '__main__':
    print(predict('data/female.wav'))
    print(predict('data/male.wav'))
