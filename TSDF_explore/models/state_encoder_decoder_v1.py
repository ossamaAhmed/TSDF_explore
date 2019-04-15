"""
 *!
 * @author    Ossama Ahmed
 * @email     oahmed@ethz.ch
 *
 * Copyright (C) 2019 Autonomous Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.asl.ethz.ch/
 *
 """

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
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

    def forward(self, x):
        padding_size = int((self.kernel_size - 1) / 2)
        if self.up_sample is not None:
            x = self.up_sample(x)
        out = F.pad(x, pad=[padding_size, padding_size, padding_size, padding_size], mode='reflect')
        out = self.conv_layer(out)
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
            BasicBlock(kernel_size=3, in_channels=32, out_channels=16),
            BasicBlock(kernel_size=3, in_channels=16, out_channels=16))

        self.encoder_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16**3, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(True)
        )

        self.bottleneck = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1024),
            nn.ReLU(True))

        self.decoder_output = nn.Sequential(
            BasicBlock(kernel_size=3, in_channels=16, out_channels=16, up_sample=True),
            BasicBlock(kernel_size=3, in_channels=16, out_channels=16, up_sample=True),
            BasicBlock(kernel_size=5, in_channels=16, out_channels=32, up_sample=True),
            BasicBlock(kernel_size=7, in_channels=32, out_channels=16, up_sample=True),
            BasicBlock(kernel_size=7, in_channels=16, out_channels=16, up_sample=True),
            BasicBlock(kernel_size=7, in_channels=16, out_channels=1, pooling=False)
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
        out = out.view(out.size(0), 16, 8, 8)
        out = self.decoder_output(out)

        # x = self.decoder(x)
        return out

