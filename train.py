import config
import collections
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from FNN import model
from torch.utils.data import DataLoader
from prepare_data import TextDataset

def get_dataset():
    train_data = TextDataset(config.TRAIN_DIR, transform=None)
    dev_data = TextDataset(config.DEV_DIR, transform=None)
    return train_data, dev_data

def train():
    train_data, dev_data = get_dataset()
    train_dataloader = DataLoader(train_data, num_workers=3, sampler=None)
    dev_dataloader = DataLoader(dev_data, num_workers=3, sampler=None)

    _model = model.FNN(50, 200)

    _model.training = True
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(_model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss_hist = collections.deque(maxlen=500)

    _model.train()

    print('Number of training sentences: {}'.format(len(train_data)))

    for epoch_num in range(config.EPOCHS):
        _model.train()
        epoch_loss = []

        for iter_num, data in enumerate(train_dataloader):
            input = data['input'].float()
            label = torch.squeeze(data['category'], dim=1)

            optimizer.zero_grad()
            output = _model(input)

            loss = criterion(output, label)
            if bool(loss == 0):
                continue
            loss.backward()

            torch.nn.utils.clip_grad_norm_(_model.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(loss.item())
        
            # print(
            #     'Epoch: {} | Iteration: {} | Running Loss: {}'.format(
            #         epoch_num, iter_num, np.mean(loss_hist)
            #     )
            # )
            del loss
        
        print("Evaluating dev set")

        _model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for iter_num, data in enumerate(dev_dataloader):
                input = data['input'].float()
                label = torch.squeeze(data['category'], dim=1)

                output = _model(input)
                loss = criterion(output, label)
                val_loss += loss.item()

                preds = (output > 0.5).float()

                correct += (preds == label).sum().item()
                total += len(label)

            val_loss /= len(dev_dataloader)
            val_acc = correct / total
            epoch_loss.append(val_loss)
            print(f'Epoch [{epoch_num + 1}/{config.EPOCHS}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Correct prediction: {correct}/{total}')
        
        scheduler.step(np.mean(epoch_loss))
    _model.eval()

if __name__ == "__main__":
    train()