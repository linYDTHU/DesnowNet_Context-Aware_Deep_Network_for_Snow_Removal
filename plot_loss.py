import torch
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

if __name__ == '__main__':
    fontbd = FontProperties(fname=r'c:\windows\fonts\timesbd.ttf', size=14)
    font = FontProperties(fname=r'c:\windows\fonts\times.ttf',size=14)
    checkpoint = torch.load('./checkpoints_ite100000.pth',map_location='cuda:0')
    loss_window = checkpoint['loss_window']
    loss_window = torch.log10(torch.stack(loss_window).reshape(1000, 100).mean(dim=1)).cpu()
    step = torch.linspace(1,100001,1000)
    plt.plot(step, loss_window, lw=1.5)
    plt.xlabel('Iterations', fontproperties=font)
    plt.ylabel('$\log_{10}(Loss)$', fontproperties=font)
    plt.show()
