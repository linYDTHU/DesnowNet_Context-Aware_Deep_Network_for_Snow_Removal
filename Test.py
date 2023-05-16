import os
import sys
import argparse
import torch
from torchvision import transforms

sys.path.append('./network')
from dataset import snow_dataset
from general import sort_nicely
from network.DesnowNet import DesnowNet

from metrics import psnr, ssim

sys.path.append('./network')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Test')

    argparser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )

    argparser.add_argument(
        '-root',
        type=str,
        help='root path for test'
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

    # prepare dataset
    gt_root = os.path.join(args.root, 'gt')
    mask_root = os.path.join(args.root, 'mask')
    synthetic_root = os.path.join(args.root, 'synthetic')
    dataset = snow_dataset(gt_root, mask_root, synthetic_root, is_crop=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=5,
                                              shuffle=True,
                                              num_workers=12,
                                              pin_memory=True)

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
    with torch.no_grad():
        psnr_sum = 0
        ssim_sum = 0
        psnr_sum_1 = 0
        ssim_sum_1 = 0
        psnr_sum_2 = 0
        ssim_sum_2 = 0
        psnr_sum_3 = 0
        ssim_sum_3 = 0
        for index, data in enumerate(data_loader):
            gt, mask, synthetic = data
            gt, mask, synthetic = gt.to(device=args.device), mask.to(device=args.device), \
                                  synthetic.to(device=args.device)
            psnr_sum += psnr(synthetic, gt)
            ssim_sum += ssim(synthetic, gt)
            y_hat, y_, z_hat, a = net(synthetic)
            y_hat[y_hat >= 1] = 1
            y_hat[y_hat <= 0] = 0
            psnr_sum_1 += psnr(y_hat, gt)
            ssim_sum_1 += ssim(y_hat, gt)
            psnr_sum_2 += psnr(y_, gt)
            ssim_sum_2 += ssim(y_, gt)
            psnr_sum_3 += psnr(z_hat, mask)
            ssim_sum_3 += ssim(z_hat, mask)
            if index >= 400:
                break
    psnr_mean = psnr_sum / 400
    ssim_mean = ssim_sum / 400
    psnr_mean_1 = psnr_sum_1 / 400
    ssim_mean_1 = ssim_sum_1 / 400
    psnr_mean_2 = psnr_sum_2 / 400
    ssim_mean_2 = ssim_sum_2 / 400
    psnr_mean_3 = psnr_sum_3 / 400
    ssim_mean_3 = ssim_sum_3 / 400
    print('x  psnr:{:.4f},ssim:{:.4f}'.format(psnr_mean.data, ssim_mean.data))
    print('y_hat  psnr:{:.4f},ssim:{:.4f}'.format(psnr_mean_1.data, ssim_mean_1.data))
    print('y_  psnr:{:.4f},ssim:{:.4f}'.format(psnr_mean_2.data, ssim_mean_2.data))
    print('z_hat  psnr:{:.4f},ssim:{:.4f}'.format(psnr_mean_3.data, ssim_mean_3.data))
    print("finished")
