import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):
    def __init__(self, inplanes, outplanes, stride) -> None:
        super().__init__()
        self.conv1a = nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(
            outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1a(self.conv1a(x)))
        out = self.conv2a(out)
        if self.stride == 1:
            residual = x
        else:
            residual = self.downsample(x)
        out += residual
        residual = out
        out = F.relu(self.outbna(out))

        out = F.relu(self.bn1b(self.conv1b(out)))
        out = self.conv2b(out)
        out += residual
        out = F.relu(self.outbnb(out))

        return out


class ResNet(nn.Module):
    def __init__(self, inplanes, outplanes) -> None:
        super().__init__()
        assert inplanes <= outplanes // 8, "inplanes must be <= outplanes//8"
        self.layer1 = ResNetLayer(inplanes, outplanes // 8, stride=1)
        self.layer2 = ResNetLayer(outplanes // 8, outplanes // 4, stride=2)
        self.layer3 = ResNetLayer(outplanes // 4, outplanes // 2, stride=2)
        self.layer4 = ResNetLayer(outplanes // 2, outplanes, stride=2)
        # self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.inplanes = inplanes
        # self.outplanes = outplanes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out


class ResNetVideoEncoder(nn.Module):
    def __init__(self, inplanes = 1, outplanes = 32) -> None:
        super().__init__()
        self.frontend3D = nn.Sequential(  # head 3d frontend
            nn.Conv3d(
                inplanes,
                outplanes // 8,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(outplanes // 8, eps=0.001, momentum=0.01),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.resnet = ResNet(outplanes // 8, outplanes)
        self.outplanes = outplanes

        # initialize weights
        self._initialize_weights()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).transpose(1, 2)
        b = x.shape[0]
        out = self.frontend3D(x)

        out = out.transpose(1, 2)
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3], out.shape[4])
        out = self.resnet(out)

        out = out.reshape(b, -1, self.outplanes)
        out = out.transpose(1, 2)
        out = out.transpose(1, 2).transpose(0, 1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
