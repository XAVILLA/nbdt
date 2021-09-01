"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbdt.models.utils import get_pretrained_model
import torchvision


__all__ = ("ResNet10", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "ResNet50_pytorch")


model_urls = {
    (
        "ResNet10",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet10.pth",
    (
        "ResNet10",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet10.pth",
    (
        "ResNet18",
        "CIFAR10",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR10-ResNet18.pth",
    (
        "ResNet18",
        "CIFAR100",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-CIFAR100-ResNet18.pth",
    (
        "ResNet18",
        "TinyImagenet200",
    ): "https://github.com/alvinwan/neural-backed-decision-trees/releases/download/0.0.1/ckpt-TinyImagenet200-ResNet18.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[2:])  # global average pooling
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.features(x)
        out = self.linear(out)
        return out


def _ResNet(arch, *args, pretrained=False, progress=True, dataset="CIFAR10", **kwargs):
    model = ResNet(*args, **kwargs)
    model = get_pretrained_model(
        arch, dataset, model, model_urls, pretrained=pretrained, progress=progress
    )
    return model


def ResNet10(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet10",
        BasicBlock,
        [1, 1, 1, 1],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet18(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet34(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet50(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet101(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet152(pretrained=False, progress=True, **kwargs):
    return _ResNet(
        "ResNet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def ResNet50_pytorch(pretrained=False, progress=True, **kwargs):
    net = torchvision.models.resnet50(pretrained=True, progress=True, **kwargs)
    # for param in net.parameters():
    #     param.requires_grad = False
    # net.fc = nn.Sequential(
    #     nn.Linear(2048, 65),
    #     # nn.ReLU(inplace=True),
    #     # nn.Linear(128, 65)
    # )
    return net


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()
