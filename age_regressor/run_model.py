import torch
from utils.preprocess import timit_dataloader
from gender_detector.model import Model
import librosa
from utils.preprocess import timit_dataloader, dataloader
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} mode')
model = Model()
model.load_state_dict(
    torch.load(
        "age_regressor/checkpoint/best.pt", map_location={'cuda:0': 'cpu'}))

if device == torch.device('cuda'):
    model.cuda()
else:
    model.cpu()
model.eval()


def predict(wavfile):
    _timit_dataloader = timit_dataloader(train_mode=False)
    waveform, _ = librosa.load(wavfile, sr=16000)

    input = _timit_dataloader.preprocess_sample_inference(waveform)
    with torch.no_grad():
        out = model(input).cpu().detach().numpy()
    out  = [round(tmp) for tmp in out]
    return out


if __name__ == '__main__':
    _timit_dataloader = timit_dataloader(data_path='./data/TIMIT', train_mode=False, age_mode=True)
    test = _timit_dataloader.return_test()
    # print(predict('data/female.wav'))
    # print(predict('data/male.wav'))
    testset = dataloader(*test)
    model.eval()

    with torch.no_grad():
        input, target = testset.next_batch(150, device=device)
        out = model(input)
        out, target = out.cpu().detach().numpy(), target.cpu().detach().numpy()
        out = [round(tmp) for tmp in out]
        print(out, target)
        print(f'accuracy_score:{accuracy_score(out, target)}')