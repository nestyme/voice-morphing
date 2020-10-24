import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.preprocess import TimitDataset, TimitDataloader
from sklearn.metrics import accuracy_score
from gender_detector.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} mode')
patience = 500
best_loss = 1000
cnt = 0


def get_class(score):
    if score < 0.4:
        return 0
    if 0.7 < score:
        return 2
    else:
        return 1


if __name__ == '__main__':
    # os.mkdir('age_regressor/checkpoint')
    model = Model()
    if device == torch.device('cuda'):
        model.cuda()
    else:
        model.cpu()
    model.train()

    _timit_dataloader = TimitDataset(data_path='./data/TIMIT', age_mode=True)
    train, valid, test = _timit_dataloader.return_datasets()
    trainset = TimitDataloader(*train)
    validset = TimitDataloader(*valid)
    BATCH_SIZE = 64

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.999), eps=1e-5
    )

    for i in tqdm(range(5000)):

        optimizer.zero_grad()

        input, target = trainset.next_batch(BATCH_SIZE, device=device)
        out = model(input)
        loss = model.loss(out, target)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            model.eval()

            with torch.no_grad():
                optimizer.zero_grad()

                input, target = validset.next_batch(BATCH_SIZE, device=device)
                out = model(input)
                valid_loss = model.loss(out, target)
                out, target = out.cpu().detach().numpy(), target.cpu().detach().numpy()
                # out = [tmp*80 for tmp in out]
                # target = [tmp*80 for tmp in target]
                print(list(zip(target, out)))
                print(f'accuracy_score:{accuracy_score([get_class(tmp) for tmp in out], [get_class(tmp) for tmp in target])}')
                print("i {}, valid {}".format(i, valid_loss.item()))
                print("_________")

            model.train()

        if i % 50 == 0 and best_loss > valid_loss.item():
            print('new best')
            best_loss = valid_loss.item()
            torch.save(model.state_dict(), "age_regressor/checkpoint/best.pt".format(i))
            cnt = 0
        else:
            cnt += 1

        if cnt > patience:
            break
    print('training finished')
