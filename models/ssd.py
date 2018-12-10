"""
Based on code from https://github.com/fastai/fastai/blob/master/courses/dl2/pascal-multi.ipynb
"""

import torch.nn as nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout=0.1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_labels, k, classification_bias):
        super().__init__()

        self.conv_classification = nn.Conv2d(in_channels, (num_labels + 1) * k,
                                             kernel_size=3, stride=1, padding=1)
        self.conv_classification.bias.data.zero_().add_(classification_bias)
        self.conv_localization = nn.Conv2d(in_channels, 4 * k, kernel_size=3, stride=1, padding=1)
        self.k = k

    def forward(self, x):
        # TODO: To combine outputs of a few layers of the feature pyramid, flatten each of them
        # and concatenate results.
        # Anchor boxes will have to follow the order of pyramid layers
        # TODO: Find a way to write a unit test that checks that structure of the model reflects anchor box coordinates
        class_conv = self.conv_classification(x)
        class_flattened = self.flatten(class_conv)

        loc_conv = self.conv_localization(x)
        loc_flattened = self.flatten(loc_conv)

        return [loc_flattened, class_flattened]

    def flatten(self, x):
        batch_size, num_features, height, width = x.size()
        # Permute to shape: (batch_size, height, width, num_features)
        x = x.permute(0, 2, 3, 1).contiguous()
        # Reshape to shape (batch_size, k*w*h, outputs_per_anchor)
        return x.view(batch_size, -1, num_features // self.k)


class SSDHead(nn.Module):
    def __init__(self, k, num_labels, classification_bias, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv1 = Conv(512, 256, 1)
        # self.conv2 = Conv(256, 256, 2)
        self.conv3 = Conv(256, 256, 2)
        self.out_conv = OutConv(256, num_labels, k, classification_bias)

    def forward(self, x):
        # Expected input shape: (num_batches, 512, 7, 7) after resnet34 backbone
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv1(x)  # Out shape: (num_batches, 256, 7, 7)
        # x = self.conv2(x)
        x = self.conv3(x)  # Out shape: (num_batches, 256, 4, 4)
        x = self.out_conv(x)  # Out shape: (num_batches, 16 * k, 4), (num_batches, 16 * k, num_labels + 1)
        return x


class SSDModel(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.head = head

    def forward(self, x):
        x = self.base_model(x)
        print(x.size())
        x = self.head(x)
        return x
