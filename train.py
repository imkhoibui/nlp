import config
import collections
import numpy as np
import torch
import torch.optim as optim

from RNN import model, eval, loss
from torch.utils.data import DataLoader
from prepare_data import TextDataset, collater

def get_dataset():
    train_data = TextDataset(config.TRAIN_DIR)
    dev_data = TextDataset(config.DEV_DIR)
    return train_data, dev_data

def train():
    train_data, dev_data = get_dataset()
    train_dataloader = DataLoader(train_data, num_workers=3, sampler=None)

    rnn = model.ClassificationModel(num_classes=train_data.num_classes())

    rnn.training = True
    
    optimizer = optim.Adam(rnn.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss_hist = collections.deque(maxlen=500)

    rnn.train()

    print('Number of training sentences: {}'.format(len(train_data)))

    for epoch_num in range(config.EPOCHS):
        rnn.train()
        epoch_loss = []

        for iter_num, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            loss = rnn(data['input'].float(), data['category'])
            if bool(loss == 0):
                continue
            loss.backward()

            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(loss)
        
            print(
                'Epoch: {} | Iteration: {} | Running Loss: {}'.format(
                    epoch_num, iter_num, np.mean(loss_hist)
                )
            )

            del loss
        
        print("Evaluating dev set")
        scores = eval.evaluate(dev_data, rnn)


        scheduler.step(np.mean(epoch_loss))
    rnn.eval()


    return

if __name__ == "__main__":
    train()