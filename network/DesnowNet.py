import torch
import torch.nn as nn
from Descriptor import Descriptor
from Recovery_Submodule import R_t, Pyramid_maxout


class TR(nn.Module):
    # translucency recovery(TR) module
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.R_t = R_t(385, beta)

    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        y_, f_c, z_hat, a = self.R_t(x, f_t, **kwargs)
        return y_, f_c, z_hat, a


class TR_new(nn.Module):
    # A new translucency recovery(TR) module with two descriptors
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_new, self).__init__()
        self.D_t_1 = Descriptor(input_channel, gamma)
        self.D_t_2 = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.AE = Pyramid_maxout(385, 3, beta)

    def forward(self, x, **kwargs):
        f_t_1 = self.D_t_1(x)
        z_hat = self.SE(f_t_1)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        z_hat_ = z_hat.detach()
        f_t_2 = self.D_t_2(x)
        a = self.AE(f_t_2)
        # yield estimated snow-free image y'
        y_ = (z_hat_ < 1) * (x - a * z_hat_) / (1 - z_hat_ + 1e-8) + (z_hat_ == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat_, a], dim=1)
        return y_, f_c, z_hat, a

class TR_za(nn.Module):
    # A  translucency recovery(TR) module predict z\times a
    def __init__(self, input_channel=3, beta=4, gamma=4):
        super(TR_za, self).__init__()
        self.D_t = Descriptor(input_channel, gamma)
        self.SE = Pyramid_maxout(385, 1, beta)
        self.SAE = Pyramid_maxout(385, 3, beta)

    def forward(self, x, **kwargs):
        f_t = self.D_t(x)
        z_hat = self.SE(f_t)
        za = self.SAE(f_t)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        za[za >= 1] = 1
        za[za <= 0] = 0
        # yield estimated snow-free image y'
        y_ = (z_hat < 1) * (x - za) / (1 - z_hat + 1e-8) + (z_hat == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        f_c = torch.cat([y_, z_hat, za], dim=1)
        return y_, f_c, z_hat, za

class RG(nn.Module):
    # the residual generation (RG) module
    def __init__(self, input_channel=7, beta=4, gamma=4):
        super(RG, self).__init__()
        self.D_r = Descriptor(input_channel, gamma)
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(385, 3, 2 * i + 1, 1, padding=i))
        self.conv_module = nn.ModuleList(block)
        self.activation = nn.Tanh()

    def forward(self, f_c):
        f_r = self.D_r(f_c)
        for i, module in enumerate(self.conv_module):
            if i == 0:
                r = module(f_r)
            else:
                r += r + module(f_r)
        r = self.activation(r)
        return r


class DesnowNet(nn.Module):
    # the DesnowNet
    def __init__(self, input_channel=3, beta=4, gamma=4, mode='original'):
        super(DesnowNet, self).__init__()
        if mode == 'original':
            self.TR = TR(input_channel, beta, gamma)
        elif mode == 'new_descriptor':
            self.TR = TR_new(input_channel, beta, gamma)
        elif mode == 'za':
            self.TR = TR_za(input_channel, beta, gamma)
        else:
            raise ValueError("Invalid architectural mode")
        self.RG = RG(beta=beta, gamma=gamma)

    def forward(self, x, **kwargs):
        y_, f_c, z_hat, a = self.TR(x, **kwargs)
        r = self.RG(f_c)
        y_hat = r + y_
        return y_hat, y_, z_hat, a


if __name__ == '__main__':
    device = 'cuda'
    net = DesnowNet().to(device)
    mask = torch.zeros([2, 1, 64, 64]).to(device)
    img = torch.zeros([2, 3, 64, 64]).to(device)
    y_hat, y_, z_hat, a = net(img, mask=mask)
    y_hat.mean().backward()
    print("finished")
