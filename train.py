import sys
import getopt
import argparse
import os
import shutil
import time
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from dataset import *
best_acc1 = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0，1"

img_root = '/home/jxk/Dataset/'
train_class = read_pickle_file('./train.pickle')
valid_class = read_pickle_file('./valid.pickle')

def main(argv):
    inputfile = ''
    savefile = './model.pth'
    testfile = ''

    print("the input datasets is :", inputfile)
    print("the test datasets is :", testfile)
    print("the save model file is :", savefile)
    model = models.resnet18(pretrained=True)

    fc_features = model.fc.in_features

    model.fc = nn.Linear(fc_features, 102)
    model = torch.nn.DataParallel(model).cuda()

    # parameters
    arch = 'resnet18'
    lr = 0.05
    momentum = 0.9
    weight_decay = 1e-4
    resume = ''
    epochs = 30
    start_epoch = 0
    evaluate = 0
    best_prec1 = 0
    print_freq = 10
    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    cudnn.benchmark = True
    #     data prepearing
    train_dir = inputfile

    valid_dir = testfile
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    batch_size = 10
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # TODO: Load the datasets with ImageFolder
    train_dataset = Flower(img_root=img_root, file_class=train_class, img_transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               pin_memory=True)

    valid_dataset = Flower(img_root=img_root, file_class=valid_class, img_transform=data_transforms)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=False,
                                                pin_memory=True)


    # train_datasets = dsets.ImageFolder(train_dir,
    #                                    data_transforms)
    #
    # # TODO: Using the image datasets and the trainforms, define the dataloaders
    # trainloader = torch.utils.data.DataLoader(dataset=train_datasets,
    #                                           batch_size=batch_size,
    #                                           shuffle=True,
    #                                           drop_last=False,
    #                                           pin_memory=True)
    # valid_datasets = dsets.ImageFolder(valid_dir,
    #                                    data_transforms)
    # validloader = torch.utils.data.DataLoader(dataset=valid_datasets,
    #                                           batch_size=batch_size,
    #                                           shuffle=True,
    #                                           drop_last=False,
    #                                           pin_memory=True)
    #     training
    best_acc1 = 0
    for epoch in range(start_epoch, epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(valid_loader, model, criterion)
        best_acc1 = max(prec1, best_acc1)
    #         saving
    save_checkpoint(model, savefile)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_freq = 10
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        target = Variable(target).cuda()
        input = Variable(input).cuda()
        # target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.avg:.3f}\t'
                  'Prec@5 {top5.avg:.3f}'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_freq = 10
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1  {top1.avg:.3f}\t'
                      'Prec@5  {top5.avg:.3f}'.format(
                    i, len(val_loader), loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(model, filename):
    torch.save(model, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main(sys.argv[1:])