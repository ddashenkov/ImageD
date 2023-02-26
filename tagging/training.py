import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from dataset import read_train
from tags import attach_tags
import model
from tagging.dataset import TaggingDataset
from torch.utils.data import DataLoader


def _loader() -> DataLoader:
    data = attach_tags(read_train())
    ds = TaggingDataset(data)
    return DataLoader(ds, batch_size=16, drop_last=True)


def _epoch(dataloader, model, loss_fn, optimizer, max_batches):
    for batch, (X, y) in enumerate(dataloader):
        if max_batches is not None and batch >= max_batches:
            break
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 1000 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}  [{current:>5d}")


def train(epochs=1, max_batches=None) -> model.TaggingModel:
    m = model.model()
    m.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=1e-3)
    for epoch in range(epochs):
        print('Training epoch ' + str(epoch))
        _epoch(_loader(), m, loss_fn, optimizer, max_batches)
    return m


if __name__ == '__main__':
    train(1, 10)
