import torch.nn as nn


class ResidalLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel_1,
        out_channel_2,
        out_channel_3,
        stride=1,
    ):
        super().__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel_3 = out_channel_3
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel, out_channels=out_channel_1, kernel_size=(1, 1)
            ),
            nn.BatchNorm2d(out_channel_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channel_1,
                out_channels=out_channel_2,
                kernel_size=(3, 3),
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channel_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channel_2,
                out_channels=out_channel_3,
                kernel_size=(1, 1),
            ),
            nn.BatchNorm2d(out_channel_3),
        )

        self.ii_downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel_3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channel_3),
        )

    def forward(self, x):
        identity = x.clone()
        m = self.conv(x)
        if self.stride != 1 or self.in_channel != self.out_channel_3:
            identity = self.ii_downsample(identity)
        return nn.ReLU(inplace=True)(m + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            ResidalLayer(
                in_channel=64,
                out_channel_1=64,
                out_channel_2=64,
                out_channel_3=256,
            ),
            ResidalLayer(
                in_channel=256,
                out_channel_1=64,
                out_channel_2=64,
                out_channel_3=256,
            ),
            ResidalLayer(
                in_channel=256,
                out_channel_1=64,
                out_channel_2=64,
                out_channel_3=256,
            ),
        )
        self.conv2 = nn.Sequential(
            ResidalLayer(
                in_channel=256,
                out_channel_1=128,
                out_channel_2=128,
                out_channel_3=512,
                stride=2,
            ),
            ResidalLayer(
                in_channel=512,
                out_channel_1=128,
                out_channel_2=128,
                out_channel_3=512,
            ),
            ResidalLayer(
                in_channel=512,
                out_channel_1=128,
                out_channel_2=128,
                out_channel_3=512,
            ),
            ResidalLayer(
                in_channel=512,
                out_channel_1=128,
                out_channel_2=128,
                out_channel_3=512,
            ),
        )
        self.conv3 = nn.Sequential(
            ResidalLayer(
                in_channel=512,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
                stride=2,
            ),
            ResidalLayer(
                in_channel=1024,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
            ),
            ResidalLayer(
                in_channel=1024,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
            ),
            ResidalLayer(
                in_channel=1024,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
            ),
            ResidalLayer(
                in_channel=1024,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
            ),
            ResidalLayer(
                in_channel=1024,
                out_channel_1=256,
                out_channel_2=256,
                out_channel_3=1024,
            ),
        )

        self.conv4 = nn.Sequential(
            ResidalLayer(
                in_channel=1024,
                out_channel_1=512,
                out_channel_2=512,
                out_channel_3=2048,
                stride=2,
            ),
            ResidalLayer(
                in_channel=2048,
                out_channel_1=512,
                out_channel_2=512,
                out_channel_3=2048,
            ),
            ResidalLayer(
                in_channel=2048,
                out_channel_1=512,
                out_channel_2=512,
                out_channel_3=2048,
            ),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ffn = nn.Linear(2048, num_classes)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.max_pool(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg_pool(x)
        avg = self.flatten(x)
        x = self.ffn(avg)

        return x, avg
