"""
Object Detection training and inference code
Adapted from: https://github.com/pytorch/vision/tree/master/references/detection
"""

import torch
import torchvision
import os
import test2
import datetime
import argparse
import time
import math
import sys
import detection_util
from torch.utils import data
from torchvision import transforms
from svhn_detect_data import SVHNFull
from networks.resnet_big import SupConResNet, ResNetDetection, SupResNetDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# os.environ["CUDA_VISIBLE_DEVICES"]= ""


def train(model, optimizer, loader, device, epoch, print_freq):
    model.train()
    metric_logger = detection_util.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', detection_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(loader) -1)

        lr_scheduler = detection_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(loader, print_freq, header):

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = detection_util.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = detection_util.MetricLogger(delimiter="  ")
    header = 'Test:'
    tp_total, fp_total, fn_total = torch.zeros([26, 10]), torch.zeros([26, 10]), torch.zeros([26, 10])

    for images, targets in metric_logger.log_every(loader, 100, header):

        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        tp, fp, fn = test2.getnum_tp_fp_fn(targets, outputs)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("AP:", test2.get_mAP(tp_total, fp_total, fn_total))
    torch.set_num_threads(n_threads)
    return



def main(args):
    detection_util.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    torch.multiprocessing.set_sharing_strategy('file_system')

    #train loader
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([transforms.ToTensor(),
                                          normalize])

    train_dataset = SVHNFull(root='./datasets/SVHN_full',
                              transform=train_transform,
                              download=False)
    test_dataset = SVHNFull(root='./datasets/SVHN_full',
                             split='test',
                             transform=train_transform,
                             download=False)
    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
        test_sampler = data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None
    print(args.batch_size)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                   collate_fn=train_dataset.collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
                                  num_workers=args.workers, pin_memory=True, sampler=test_sampler,
                                  collate_fn=test_dataset.collate_fn)

    pre_mod = ResNetDetection(name='resnet18')
    #pre_mod = SupResNetDetection(name='resnet18')

    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location='cpu')
        state_dict = ckpt['model']

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        pre_mod.load_state_dict(state_dict)

    backbone = pre_mod.encoder
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((16, 32, 128),), aspect_ratios=((0.5, 1.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=3, sampling_ratio=2)

    model = FasterRCNN(backbone, num_classes=10, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, rpn_batch_size_per_image=64)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1


    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(model, optimizer, train_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        #if epoch > 15:
        evaluate(model, test_loader, device=device)
        detection_util.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'args': args,
            'epoch': epoch},
            os.path.join('save/detector/', 'model_{}.pth'.format(epoch)))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int, help='images per gpu')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('-epochs', default=50, type=int, help='number of epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr-steps', default=[30, 40], nargs='+', type=int, help='decrease lr every n steps')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='lr decrease factor')
    parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
    parser.add_argument('--aspect-ratio-group-factor', default=2, type=int)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='', help='load pretrained model')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    args = parser.parse_args()

    main(args)
