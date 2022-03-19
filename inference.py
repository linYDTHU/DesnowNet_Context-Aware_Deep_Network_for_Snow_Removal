import os
import sys
import argparse
import torch
from torchvision import transforms

sys.path.append('./network')
from general import sort_nicely
from network.DesnowNet import DesnowNet

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('./network')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Inference')

    argparser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )

    argparser.add_argument(
        '-path',
        type=str,
        help='path of the image for inference'
    )

    argparser.add_argument(
        '-dir',
        type=str,
        help='path of checkpoints'
    )

    argparser.add_argument(
        '--checkpoint',
        type=int,
        help='choose a checkpoint'
    )

    argparser.add_argument(
        '-beta',
        type=int,
        default=4,
        help='the scale of the pyramid maxout'
    )

    argparser.add_argument(
        '-gamma',
        type=int,
        default=4,
        help='the levels of the dilation pyramid'
    )

    argparser.add_argument(
        '--mode',
        type=str,
        default='original',
        help='the architectural mode of DesnowNet'
    )



    args = argparser.parse_args()

    # read the image
    toTensor = transforms.ToTensor()
    img_data = Image.open(args.path).convert('RGB')
    mask_data = Image.open("../data/Test/Snow100K-L/mask/beautiful_smile_00003.jpg").convert('L')
    gt_data = Image.open("../data/Test/Snow100K-L/gt/beautiful_smile_00003.jpg")

    img_tensor = toTensor(img_data).unsqueeze(0).to(device=args.device)
    mask_tensor = toTensor(mask_data).unsqueeze(0).to(device=args.device)
    gt_tensor = toTensor(gt_data).unsqueeze(0).to(device=args.device)
    with torch.no_grad():
        a_gt = (img_tensor - (1 - mask_tensor) * gt_tensor) / (1e-8 + mask_tensor) * (mask_tensor != 0)

    # load checkpoint
    net = DesnowNet(beta=args.beta, gamma=args.gamma, mode=args.mode).to(args.device)
    checkpoint_files = os.listdir(args.dir)
    if checkpoint_files:
        checkpoint_name = "checkpoints_ite{}.pth".format(args.checkpoint)
        checkpoint = torch.load(os.path.join(args.dir, checkpoint_name))
        net.load_state_dict(checkpoint['state_dict'])
        print("load checkpoints(iteration={})".format(checkpoint['iteration']))
    else:
        raise RuntimeError('No checkpoint in logs directory')

    net.eval()
    ToPILImage = transforms.ToPILImage()
    with torch.no_grad():
        y_hat, y_, z_hat, a = net(img_tensor)
        y_hat = (y_hat > 1).float() + (1 >= y_hat) * (y_hat >= 0) * y_hat
        y_ = (y_ > 1).float() + (1 >= y_) * (y_ >= 0) * y_
    img_1 = ToPILImage(y_hat.cpu().squeeze(0))
    img_2 = ToPILImage(y_.cpu().squeeze(0))
    img_3 = ToPILImage(z_hat.cpu().squeeze(0))
    plt.subplot(2,2,1)
    plt.imshow(img_data)
    plt.xticks([])
    plt.yticks([])
    plt.title('(a)', y=-0.15)
    plt.subplot(2,2,2)
    plt.imshow(img_1)
    plt.xticks([])
    plt.yticks([])
    plt.title('(b)', y=-0.15)
    plt.subplot(2,2,3)
    plt.imshow(img_2)
    plt.xticks([])
    plt.yticks([])
    plt.title('(c)', y=-0.15)
    plt.subplot(2,2,4)
    plt.imshow(img_3,cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.title('(d)', y=-0.15)
    # plt.subplot(1,3,1)
    # plt.imshow(img_data)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('(a)', y=-0.25)
    # plt.subplot(1,3,2)
    # plt.imshow(mask_data, cmap=plt.cm.gray)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('(b)', y=-0.25)
    # plt.subplot(1,3,3)
    # plt.imshow(img_3, cmap=plt.cm.gray)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('(c)', y=-0.25)
    plt.subplots_adjust(0.05,0.05,0.95,0.95)
    plt.show()

    print("finished")
