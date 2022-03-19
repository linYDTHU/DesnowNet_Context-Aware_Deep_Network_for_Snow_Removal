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
from network.DesnowNet import DesnowNet, TR

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
        '--TR_iterations',
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

    args = argparser.parse_args()

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
    if not os.path.exists(os.path.join(args.dir, "TR")):
        os.mkdir(os.path.join(args.dir, "TR"))
    if not os.path.exists(os.path.join(args.dir, "net")):
        os.mkdir(os.path.join(args.dir, "net"))

    TR_dir = os.path.join(args.dir, "TR")
    net_dir = os.path.join(args.dir, "net")

    # load TR_checkpoint
    TR_checkpoint_files = os.listdir(TR_dir)
    if TR_checkpoint_files:
        sort_nicely(TR_checkpoint_files)
        TR_latest_checkpoint = TR_checkpoint_files[-1]
        TR_checkpoint = torch.load(os.path.join(TR_dir, TR_latest_checkpoint))
        TR_iteration = TR_checkpoint['iteration']
    else:
        TR_iteration = 0

    ## Train TR module
    TR_module = TR(beta=args.beta, gamma=args.gamma).to(args.device)
    TR_optimizer = optim.Adam(TR_module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if not TR_checkpoint_files:
        # initialization
        for name, param in TR_module.named_parameters():
            if 'conv.weight' in name and 'bn' not in name and 'activation' not in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0.0)
    else:
        TR_module.load_state_dict(TR_checkpoint['state_dict'])
        TR_optimizer.load_state_dict(TR_checkpoint['optimizer'])

    TR_module.train()
    while TR_iteration < args.TR_iterations:
        for data in data_loader:
            TR_iteration += 1

            gt, mask, synthetic = data
            gt, mask, synthetic = gt.to(device=args.device), mask.to(device=args.device), \
                                  synthetic.to(device=args.device)
            # with torch.no_grad():
            #     a_gt = ((synthetic - (1 - mask) * gt) / (1e-8 + mask) * (mask != 0)).to(device=args.device)
            TR_optimizer.zero_grad()
            y_, _, z_hat, a = TR_module(synthetic)
            loss1 = lw_pyramid_loss(z_hat, mask)
            loss2 = lw_pyramid_loss(y_, gt)
            loss = loss2 + args.weight_mask * loss1
            loss.backward()
            TR_optimizer.step()

            print("TR Training... Iteration: %d  Loss: %f" % (TR_iteration, loss.data))

            if TR_iteration in args.save_schedule:
                state = {
                    'iteration': TR_iteration,
                    'state_dict': TR_module.state_dict(),
                    'optimizer': TR_optimizer.state_dict()
                }
                torch.save(state, os.path.join(TR_dir, 'checkpoints_ite{}.pth'.format(TR_iteration)))

            if TR_iteration >= args.TR_iterations:
                break

    # load checkpoint
    checkpoint_files = os.listdir(net_dir)
    if checkpoint_files:
        sort_nicely(checkpoint_files)
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(os.path.join(net_dir, latest_checkpoint))
        iteration = checkpoint['iteration']
    else:
        iteration = 0

    #Train RG module (DesnowNet)
    net = DesnowNet(beta=args.beta, gamma=args.gamma).to(args.device)
    ## freeze some parameters
    for param in net.parameters():
        param.requires_grad = False
    for param in net.RG.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=args.learning_rate, weight_decay=args.weight_decay)
    if not checkpoint_files:
        # initialization
        for name, param in net.RG.named_parameters():
            if 'conv.weight' in name and 'bn' not in name and 'activation' not in name:
                init.xavier_normal_(param)
            if 'bias' in name:
                init.constant_(param, 0.0)
    else:
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    net.TR.load_state_dict(TR_module.state_dict())

    net.train()
    while iteration < args.iterations:
        for data in data_loader:
            iteration += 1

            gt, mask, synthetic = data
            gt, mask, synthetic = gt.to(device=args.device), mask.to(device=args.device), \
                                  synthetic.to(device=args.device)
            optimizer.zero_grad()
            y_hat, y_, z_hat, a = net(synthetic)
            loss = lw_pyramid_loss(y_hat, gt)
            loss.backward()
            optimizer.step()

            """
                Saving the model if necessary
            """

            if iteration in args.save_schedule:
                state = {
                    'iteration': iteration,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(net_dir, 'checkpoints_ite{}.pth'.format(iteration)))

            print("Net Training... Iteration: %d  Loss: %f" % (iteration, loss.data))
            if iteration >= args.iterations:
                break

    print("finished")
