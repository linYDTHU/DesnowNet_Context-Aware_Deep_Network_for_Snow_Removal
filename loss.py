import torch
import torch.nn as nn
from torch.autograd import Variable


def weight_decay_l2(loss, model, lambda_w):
    wdecay = 0
    for w in model.parameters():
        if w.requires_grad:
            wdecay = torch.add(torch.sum(w ** 2), wdecay)

    loss = torch.add(loss, lambda_w * wdecay)
    return loss


def lw_pyramid_loss(m, hat_m, tau=6):
    """
     lightweight pyramid loss function
    :param m: one image
    :param hat_m: another image of the same size
    :param tau:the level of loss pyramid, default 4
    :return: loss
    """
    batch_size = m.shape[0]
    loss = 0
    for i in range(tau + 1):
        block = nn.MaxPool2d(2**i, stride=2**i)
        p1 = block(m)
        p2 = block(hat_m)
        loss += torch.sum((p1-p2)**2)
    return loss/batch_size


if __name__ == '__main__':
    device = 'cuda'
    img1 = torch.zeros([5, 1, 64, 64], device=device).requires_grad_()
    img2 = torch.ones_like(img1, device=device).requires_grad_() * 0.1
    loss = lw_pyramid_loss(img1, img2)
    module1 = nn.Conv2d(3,128,3)
    loss = weight_decay_l2(loss, module1, 0.2)
    print("finished")