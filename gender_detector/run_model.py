import torch
from utils.preprocess import timit_dataloader
from gender_detector.model import Model
import librosa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} mode')
model = Model()
model.cpu()
model.load_state_dict(
    torch.load(
        "gender_detector/checkpoint/best.pt", map_location={'cuda:0': 'cpu'}))
model.cpu()
model.eval()


def predict(wavfile, device=device):
    _timit_dataloader = timit_dataloader(train_mode=False)
    waveform, _ = librosa.load(wavfile, sr=16000)

    input = _timit_dataloader.preprocess_sample_inference(waveform)
    with torch.no_grad():
        out = model(input).cpu().detach().numpy()
    out = 'female' if out < 0.5 else 'male'
    return out


if __name__ == '__main__':
    print(predict('data/female.wav'))
    print(predict('data/male.wav'))
