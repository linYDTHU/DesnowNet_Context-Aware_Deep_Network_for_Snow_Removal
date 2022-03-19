import torch
import torch.nn as nn
from Inceptionv4 import InceptionV4


class DP(nn.Module):
    # dilation pyramid
    def __init__(self, in_channel=768, depth=77, gamma=4):
        super(DP, self).__init__()
        self.gamma = gamma
        block = []
        for i in range(gamma + 1):
            block.append(nn.Conv2d(in_channel, depth, 3, 1, padding=2 ** i, dilation=2 ** i))
        self.block = nn.ModuleList(block)

    def forward(self, feature):
        for i, block in enumerate(self.block):
            if i == 0:
                output = block(feature)
            else:
                output = torch.cat([output, block(feature)], dim=1)
        return output


class Descriptor(nn.Module):
    def __init__(self, input_channel=3, gamma=4):
        super(Descriptor, self).__init__()
        self.backbone = InceptionV4(input_channel)
        self.DP = DP(gamma=gamma)

    def forward(self, img):
        feature = self.backbone(img)
        f = self.DP(feature)
        return f


if __name__ == '__main__':
    device = 'cpu'
    Descriptor_1 = Descriptor().to(device)
    img = torch.zeros([1, 3, 200, 200]).to(device)
    f = Descriptor_1(img)
    f.mean().backward()
    print("finished")
