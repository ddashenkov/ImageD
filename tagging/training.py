import pathlib
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from dataset import read_train
from tags import attach_tags
import model
from tagging.dataset import TaggingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def _loader() -> DataLoader:
    data = attach_tags(read_train())
    ds = TaggingDataset(data)
    return DataLoader(ds, batch_size=8, drop_last=True)


def _epoch(epoch_number, dataloader, model, loss_fn, optimizer, max_batches):
    for batch, (X, y) in enumerate(dataloader):
        if max_batches is not None and batch >= max_batches:
            break
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 4:
            loss_value, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss_value:>7f} @ {batch}")
            with open(pathlib.Path("~/ImageD/tagging/loss.csv").expanduser(), 'a+') as loss_log:
                loss_log.write(str(loss) + ',' + str(batch) + os.linesep)
    model.store_progress(path=f'~/ImageD/tagging/state-at-{epoch_number}.pt')


def train(epochs=1, max_batches=None) -> model.TaggingModel:
    m = model.model()
    m.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
    for epoch in tqdm(range(epochs), 'Training model...'):
        print('Training epoch ' + str(epoch))
        _epoch(epoch, _loader(), m, loss_fn, optimizer, max_batches)
    print('DONE.')
    return m


if __name__ == '__main__':
    train(20)
