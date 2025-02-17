import torch
from torch.nn import Module, Conv2d, Parameter, Softmax, Sequential, BatchNorm2d, ReLU


class Decoder(Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.classifier = Sequential(
            BatchNorm2d(1024),
            ReLU(inplace=True),
            Conv2d(1024, 512, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 256, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 128, 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 1, 1)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


