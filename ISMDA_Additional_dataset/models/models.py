import torch
from torch import nn

class EEGNet_ATTEN(nn.Module):
    def __init__(self, configs, DOR):
        super(EEGNet_ATTEN, self).__init__()
        afr_reduced_cnn_size = configs.afr_reduced_cnn_size
        Chans = configs.Chans
        dropoutRate = DOR
        kernLength1 = configs.kernLength1
        kernLength2 = configs.kernLength2
        kernLength3 = configs.kernLength3
        F1 = configs.F1
        D = configs.D
        expansion = configs.expansion
        F2 = F1 * D
        self.features1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength1), padding=(0, kernLength1 // 2), bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(F1),

            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1,16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength2), padding=(0, kernLength2 // 2), bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(F1),

            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1,16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength3), padding=(0, kernLength3 // 2), bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(F1),

            nn.Conv2d(F1, F2, (Chans, 1), groups=D, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1,8)),
            nn.Dropout(dropoutRate),

            SeparableConv2d(F2, F2, kernel_size=(1,16), padding=(0, 8)),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.MaxPool2d((1, 8))
        )

        self.dropout = nn.Dropout(dropoutRate)
        self.inplanes = F2*3
        self.ADR = self._make_layer(AttenBlock, afr_reduced_cnn_size, 1, expansion, 1)

    def _make_layer(self, block, planes, blocks, expansion, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=(1,1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x3 = self.features3(x)
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_concat = self.dropout(x_concat)
        x_concat = self.ADR(x_concat)
        x_concat = x_concat.view(x_concat.size(0), -1)
        return x_concat

class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        padding=self.padding, groups=self.c_in)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttenBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(AttenBlock, self).__init__()
        #SE
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1,1), stride = stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=(1,1), stride = stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(inplanes, planes*4, kernel_size=(1, 1), stride=stride)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        #DAR
        self.squeeze = inplanes // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Sequential(
            nn.Linear(inplanes, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.fc3(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out1 = self.se(out)

        out2 = self.conv_s1(out) * y[:, 0] + self.conv_s2(out) * y[:, 1] + \
                self.conv_s3(out) * y[:, 2] + self.conv_s4(out) * y[:, 3]

        out = out1 + out2
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()
        num_classes = configs.num_classes
        features_len = configs.features_len
        self.FC1 = nn.Linear(features_len, 500)
        self.elu = nn.ELU(inplace=True)
        self.FC2 = nn.Linear(500, num_classes)
        self.sf = nn.Softmax(dim=1)
    def forward(self, input):
        logits = self.FC1(input)
        logits = self.elu(logits)
        logits = self.FC2(logits)
        logits = self.sf(logits)
        return logits


