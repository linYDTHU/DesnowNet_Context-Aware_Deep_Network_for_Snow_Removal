import torch
import torch.nn as nn


class Pyramid_maxout(nn.Module):
    def __init__(self, in_channel=385, depth=3, beta=4):
        super(Pyramid_maxout, self).__init__()
        block = []
        for i in range(beta):
            block.append(nn.Conv2d(in_channel, depth, 2 * i + 1, 1, padding=i))
        self.activation = nn.PReLU(num_parameters=depth)
        self.conv_module = nn.ModuleList(block)

    def forward(self, f):
        for i, module in enumerate(self.conv_module):
            if i == 0:
                conv_result = module(f).unsqueeze(0)
            else:
                temp = module(f).unsqueeze(0)
                conv_result = torch.cat([conv_result, temp], dim=0)
        result, _ = torch.max(conv_result, dim=0)
        return self.activation(result)


class R_t(nn.Module):
    # The recovery submodule (Rt) of the translucency recovery (TR) module
    def __init__(self, in_channel=385, beta=4):
        super(R_t, self).__init__()
        self.SE = Pyramid_maxout(in_channel, 1, beta)
        self.AE = Pyramid_maxout(in_channel, 3, beta)


    def forward(self, x, f_t, **kwargs):
        z_hat = self.SE(f_t)
        a_hat = self.AE(f_t)
        z_hat[z_hat >= 1] = 1
        z_hat[z_hat <= 0] = 0
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            z = kwargs['mask']
            a = kwargs['a']
        elif 'mask' in kwargs.keys():
            z = kwargs['mask']
            a = a_hat
        else:
            z = z_hat
            a = a_hat
        # yield estimated snow-free image y'
        y_ = (z < 1) * (x - a_hat * z) / (1 - z + 1e-8) + (z == 1) * x
        y_[y_ >= 1] = 1
        y_[y_ <= 0] = 0
        # yield feature map f_c
        if 'mask' in kwargs.keys() and 'a' in kwargs.keys():
            with torch.no_grad():
                y = (z < 1) * (x - a * z) / (1 - z + 1e-8) + (z == 1) * x
            f_c = torch.cat([y, z, a], dim=1)
        elif 'mask' in kwargs.keys():
            f_c = torch.cat([y_, z, a], dim=1)
        else:
            f_c = torch.cat([y_, z, a], dim=1)
        return y_, f_c, z_hat, a_hat


if __name__ == '__main__':
    device = 'cpu'
    R_1 = R_t().to(device)
    f_t = torch.randn([1, 385, 64, 64]).to(device)
    img = torch.zeros([1, 3, 200, 200]).to(device)
    y_, f_c, z_hat, za = R_1(img, f_t)
    y_.mean().backward()
    print("finished")
