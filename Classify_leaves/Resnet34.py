import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion = 1  # 用于支持扩展性

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.left(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet34, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet34(num_classes=1000):
    return ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# def main():
#     model = resnet34(num_classes=1000)
#     x = torch.randn(1, 3, 224, 224)
#     print(f"Input shape: {x.shape}")
#     x = model.conv1(x)
#     print(f"After conv1: {x.shape}")
#     x = model.maxpool(x)
#     print(f"After maxpool: {x.shape}")

#     x = model.layer1(x)
#     print(f"After layer1: {x.shape}")
#     x = model.layer2(x)
#     print(f"After layer2: {x.shape}")
#     x = model.layer3(x)
#     print(f"After layer3: {x.shape}")
#     x = model.layer4(x)
#     print(f"After layer4: {x.shape}")

#     x = model.avgpool(x)
#     print(f"After avgpool: {x.shape}")
#     x = torch.flatten(x, 1)
#     print(f"After flatten: {x.shape}")
#     x = model.fc(x)
#     print(f"After fc: {x.shape}")

# if __name__ == '__main__':
#     main()
