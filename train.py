import os
import sys
import argparse
import torch
import torch.nn.init as init
import torch.optim as optim
from loss import weight_decay_l2, lw_pyramid_loss

sys.path.append('./network')
from dataset import snow_dataset
from general import sort_nicely
from network.DesnowNet import DesnowNet

sys.path.append('./network')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Train the model')

    argparser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )

    argparser.add_argument(
        '-r',
        '--root',
        type=str,
        help='root directory of trainset'
    )

    argparser.add_argument(
        '-dir',
        type=str,
        default='./_logs',
        help='path to store the model checkpoints'
    )

    argparser.add_argument(
        '-iter',
        '--iterations',
        type=int,
        default=2000
    )

    argparser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=3e-5
    )

    argparser.add_argument(
        '--batch_size',
        type=int,
        default=5
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
        '--weight_decay',
        type=float,
        default=5e-4
    )

    argparser.add_argument(
        '--weight_mask',
        type=float,
        default=3,
        help='the weighting to leverage the importance of snow mask'
    )

    argparser.add_argument(
        '--save_schedule',
        type=int,
        nargs='+',
        default=[],
        help='the schedule to save the model'
    )

    argparser.add_argument(
        '--mode',
        type=str,
        default='original',
        help='the architectural mode of DesnowNet'
    )

    args = argparser.parse_args()
    net = DesnowNet(beta=args.beta, gamma=args.gamma, mode=args.mode).to(args.device)

    # initialization
    for name, param in net.named_parameters():
        if 'conv.weight' in name and 'bn' not in name and 'activation' not in name:
            init.xavier_normal_(param)
            # print(name, param.data)
        if 'bias' in name:
            init.constant_(param, 0.2)
            # print(name, param.data)

    # prepare dataset
    gt_root = os.path.join(args.root, 'gt')
    mask_root = os.path.join(args.root, 'mask')
    synthetic_root = os.path.join(args.root, 'synthetic')
    dataset = snow_dataset(gt_root, mask_root, synthetic_root)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=6,
                                              pin_memory=True)

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    # load checkpoint
    checkpoint_files = os.listdir(args.dir)
    if checkpoint_files:
        sort_nicely(checkpoint_files)
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(os.path.join(args.dir, latest_checkpoint))
        iteration = checkpoint['iteration']
        best_loss = checkpoint['best_loss']
        best_loss = best_loss.to(device=args.device)
        best_loss_iter = checkpoint['best_loss_iter']
        loss_window = checkpoint['loss_window']
    else:
        iteration = 0
        best_loss = 10000000.0
        best_loss_iter = 0
        loss_window = []

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10000,verbose=True)
    if not checkpoint_files:
        # initialization
        for name, param in net.named_parameters():
            if 'conv.weight' in name and 'bn' not in name and 'activation' not in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0.0)
    else:
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    net.train()
    while iteration < args.iterations:
        for data in data_loader:
            iteration += 1

            gt, mask, synthetic = data
            gt, mask, synthetic = gt.to(device=args.device), mask.to(device=args.device), \
                                  synthetic.to(device=args.device)
            optimizer.zero_grad()
            y_hat, y_, z_hat, za = net(synthetic)
            loss1 = lw_pyramid_loss(y_hat, gt)
            if args.mode == 'za':
                with torch.no_grad():
                    za_gt = synthetic - (1-mask)*gt
                loss2 = lw_pyramid_loss(za, za_gt)
            else:
                loss2 = lw_pyramid_loss(y_, gt)
            loss3 = lw_pyramid_loss(z_hat, mask)
            loss = loss1 + loss2 + args.weight_mask * loss3
            # loss = loss3
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            """
                Saving the model if necessary
            """
            loss_window.append(loss.data)
            if loss.data < best_loss:
                best_loss = loss.data
                best_loss_iter = iteration

            if iteration in args.save_schedule:
                state = {
                    'iteration': iteration,
                    'state_dict': net.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter,
                    'loss_window': loss_window
                }
                torch.save(state, os.path.join(args.dir, 'checkpoints_ite{}.pth'.format(iteration)))

            print("Iteration: %d  Loss: %f" % (iteration, loss.data))
            if iteration >= args.iterations:
                break

    print("finished")
