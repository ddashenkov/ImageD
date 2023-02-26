import pathlib

import numpy as np
import torch

from remote_images import load_image
from tags import tags

_DEFAULT_STATE_PATH = '~/ImageD/tagging/state.pt'


def _load_pretrain() -> torch.nn.Module:
    core_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
    for m in core_model.modules():
        m.requires_grad_(False)
    return core_model


def _load_head(path) -> torch.nn.Module:
    state_path = pathlib.Path(path)
    fully_connected = torch.nn.Linear(2048, len(tags()))
    if state_path.exists():
        fully_connected.load_state_dict(torch.load(path))
    return fully_connected


class TaggingModel(torch.nn.Module):

    def __init__(self, core_model, head):
        super(TaggingModel, self).__init__()
        self.core_model = core_model
        self.head = head
        self.core_model.fc = head

    def forward(self, x):
        return self.core_model(x)

    def store_progress(self, path=_DEFAULT_STATE_PATH):
        torch.save(self.head.state_dict(), path)


def model(inference=False, path_to_state=_DEFAULT_STATE_PATH) -> TaggingModel:
    pretrain = _load_pretrain()
    head = _load_head(path_to_state)
    mdl = TaggingModel(pretrain, head)
    if inference:
        mdl.eval()
    return mdl


if __name__ == '__main__':
    m = model()
    from tagging.dataset import TaggingDataset
    ds = TaggingDataset({'000002b97e5471a0': np.zeros(601)})
    tensor = next(iter(ds))['image']
    print(m(tensor))
