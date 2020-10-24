import torch
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from tqdm import tqdm

from gender_detector.model import Model
from utils.preprocess import TimitDataset, TimitDataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {device} mode')
patience = 500
best_loss = 1000
cnt = 0

if __name__ == '__main__':
    # os.mkdir('gender_detector/checkpoint')
    model = Model()
    model.to(device)

    _timit_dataloader = TimitDataset()
    train, valid, test = _timit_dataloader.return_datasets()

    trainset = TimitDataloader(*train)
    validset = TimitDataloader(*valid)
    testset = TimitDataloader(*test)

    BATCH_SIZE = 64

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.999), eps=1e-5)

    for i in tqdm(range(1000)):
        optimizer.zero_grad()

        input, target = trainset.next_batch(BATCH_SIZE, device=device)

        out = model(input)
        loss = model.loss(out, target)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            model.eval()

            with torch.no_grad():
                input, target = validset.next_batch(BATCH_SIZE, device=device)
                out = model(input)
                valid_loss = model.loss(out, target)
                out, target = out.cpu().detach().numpy(), target.cpu().detach().numpy()
                out = [1. if tmp > 0.5 else 0 for tmp in out]

                print(f'accuracy_score:{accuracy_score(out, target)}')
                print("i {}, valid {}".format(i, valid_loss.item()))
                print("_________")

            model.train()

        if i % 50 == 0 and best_loss > valid_loss.item():
            print('new best')
            best_loss = valid_loss.item()
            torch.save(model.state_dict(), "gender_detector/checkpoint/best.pt".format(i))
            cnt = 0
        else:
            cnt += 1

        if cnt > patience:
            break

    print('training finished')
