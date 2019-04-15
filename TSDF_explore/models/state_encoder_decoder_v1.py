__author__ = 'Ossama'
# modified by Yimeng @Apr.15

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels, up_sample=False, pooling=True):
        super(BasicBlock, self).__init__()
        self.kernel_size = kernel_size
        self.up_sample = None
        self.pooling = pooling
        if up_sample:
            self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear')
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        padding_size = int((self.kernel_size - 1) / 2)
        if self.up_sample is not None:
            x = self.up_sample(x)
        x = self.conv_layer(x)
        out = F.pad(x, pad=[padding_size, padding_size, padding_size, padding_size])
        out = self.relu(out)
        if self.up_sample is None and self.pooling:
            out = self.max_pool(out)
        return out


class StateEncoderDecoder(nn.Module):

    def __init__(self):
        super(StateEncoderDecoder, self).__init__()
        self.encode_features = nn.Sequential(
            BasicBlock(kernel_size=7, in_channels=2, out_channels=16),
            BasicBlock(kernel_size=5, in_channels=16, out_channels=32),
            BasicBlock(kernel_size=3, in_channels=32, out_channels=64))

        self.encoder_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(128*128, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 128*128),
            nn.ReLU(True))

        self.decoder_output = nn.Sequential(
            BasicBlock(kernel_size=3, in_channels=64, out_channels=32, up_sample=True, pooling=False),
            BasicBlock(kernel_size=5, in_channels=32, out_channels=16, up_sample=True, pooling=False),
            BasicBlock(kernel_size=7, in_channels=16, out_channels=2, up_sample=True, pooling=False)
        )

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        return out

    def encode(self, x):
        out = self.encode_features(x)
        out = out.view(out.size(0), -1)
        out = self.encoder_output(out)
        return out

    def decode(self, x):
        out = self.bottleneck(x)
        out = out.view(out.size(0), 64, 16, 16)
        out = self.decoder_output(out)

        # x = self.decoder(x)
        return out

