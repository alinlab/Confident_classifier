###########################################################
# Image classification code based on samples from pytorch #
###########################################################
# Writer: Kimin Lee 

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import model as CNN
import generate_log as genlog
import calculate_log as callog
import True_calculate_log as true_callog
import ent_calculate_log as ent_callog
import models_from_vision as models

from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch code for ')
parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='batch size for testing')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--dataset', required=True, help='target dataset: cifar10 | mnist | imagenet | SVHN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='', help='folder to output images and model checkpoints')
parser.add_argument('--nt_dataset', required=True, help='non-target dataset: cifar10 | mnist | imagenet | SVHN')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--pre_trained_net', default='', help="path to pre trained_net")
parser.add_argument('--net_type', default='VGG13', help="Yype of Classification Nets")

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load target data: ',args.dataset)
if args.dataset == 'cifar10':
    train_loader, test_loader = data_loader.getCIFAR10(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.dataset == 'mnist':
    train_loader, test_loader = data_loader.getMNIST(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.dataset == 'svhn':
    train_loader, test_loader = data_loader.getSVHN(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

print('load non target data: ',args.nt_dataset)
if args.nt_dataset == 'cifar10':
    _, nt_test_loader = data_loader.getCIFAR10(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'mnist':
    _, nt_test_loader = data_loader.getMNIST(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'svhn':
    _, nt_test_loader = data_loader.getSVHN(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'cifar100':
    _, nt_test_loader = data_loader.getCIFAR100(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'gaussian':
    _, nt_test_loader = data_loader.getCIFAR10(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'uniform':
    _, nt_test_loader = data_loader.getCIFAR10(batch_size=args.batch_size, img_size=args.imageSize, data_root=args.dataroot, num_workers=1)

elif args.nt_dataset == 'imagenet':
    testsetout = datasets.ImageFolder(args.dataroot+"/Imagenet", transform=transforms.Compose([transforms.Scale(args.imageSize),transforms.ToTensor()]))
    nt_test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

elif args.nt_dataset == 'imagenet_resize':
    testsetout = datasets.ImageFolder(args.dataroot+"/Imagenet_resize", transform=transforms.Compose([transforms.Scale(args.imageSize),transforms.ToTensor()]))
    nt_test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

elif args.nt_dataset == 'lsun':
    testsetout = datasets.ImageFolder(args.dataroot+"/LSUN", transform=transforms.Compose([transforms.Scale(args.imageSize),transforms.ToTensor()]))
    nt_test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

elif args.nt_dataset == 'lsun_resize':
    testsetout = datasets.ImageFolder(args.dataroot+"/LSUN_resize", transform=transforms.Compose([transforms.Scale(args.imageSize),transforms.ToTensor()]))
    nt_test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

elif args.nt_dataset == 'isun':
    testsetout = datasets.ImageFolder(args.dataroot+"/iSUN", transform=transforms.Compose([transforms.Scale(args.imageSize),transforms.ToTensor()]))
    nt_test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

print('load model: ',args.pre_trained_net)

if args.net_type == 'L2':
    model = CNN.CNN_2(10)
elif args.net_type == 'L7':
    model = CNN.CNN_7(n_channel=32)
elif args.net_type == 'Lenet':
    model = CNN.LENET()
elif args.net_type == 'Resnet18':
    model = CNN.RESNET(18)
elif args.net_type == 'Resnet34':
    model = CNN.RESNET(34)
elif args.net_type == 'Resnet50':
    model = CNN.RESNET(50)
elif args.net_type == 'Resnet101':
    model = CNN.RESNET(101)
elif args.net_type == 'Googlenet':
    model = CNN.GOOGLENET()
elif args.net_type == 'Vggnet':
    model = CNN.VGGNET()
elif args.net_type == 'Densenet':
    model = CNN.DENSENET()
elif args.net_type == 'VGG11':
    model = models.vgg11()
elif args.net_type == 'VGG11_BN':
    model = models.vgg11_bn()
elif args.net_type == 'VGG13':
    model = models.vgg13()
elif args.net_type == 'VGG13_BN':
    model = models.vgg13_bn()
elif args.net_type == 'VGG16':
    model = models.vgg16()
elif args.net_type == 'VGG16_BN':
    model = models.vgg16_bn()
elif args.net_type == 'RES18':
    model = models.resnet18()
elif args.net_type == 'ALEX':
    model = models.alexnet()

if args.pre_trained_net != '':
    model.load_state_dict(torch.load(args.pre_trained_net))

if args.cuda:
    model.cuda()


def target_entropy_test():
    model.eval()
    correct = 0
    total_over_entropy = 0
    total_under_entropy = 0
    count_over = 0
    count_under = 0
    for data, target in test_loader:
        if args.dataset == 'mnist':
            data = data.expand(data.size(0), 3, args.imageSize, args.imageSize).clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.softmax(model(data))
        pred = output.data.max(1)[1] # get the index of the max log-probability
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()
        for i in range(data.size(0)):
            prediction = (output.data[i]).cpu().numpy()
            entropy = -np.sum(np.multiply(prediction, np.log(np.add(prediction,0.00000001))))
            #if (equal_flag[i] == 1).numpy():
            if (equal_flag[i] == 1):
                count_under += 1
                total_under_entropy += entropy
                #ent_log_t_over.write(str(1)+','+str(entropy)+'\n')
            else:
                count_over += 1
                total_over_entropy += entropy
                #ent_log_t_over.write(str(0)+','+str(entropy)+'\n')
    total_under_entropy /= count_under
    total_over_entropy /= count_over
    #ent_log_t_over.write('\n Over entropy: {:.4f}, Under entropy: {:.4f}, Final Accuracy: {}/{} ({:.0f}%)\n'.format(total_over_entropy, total_under_entropy, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))
    print('\n Over entropy: {:.4f}, Under entropy: {:.4f}, Final Accuracy: {}/{} ({:.2f}%)\n'.format(
        total_over_entropy, total_under_entropy, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def non_target_entropy_test():
    model.eval()
    correct = 0
    total_over_entropy = 0
    count_over = 0

    for data, target in nt_test_loader:
        if args.nt_dataset == 'mnist':
            data = data.expand(data.size(0), 3, args.imageSize, args.imageSize).clone()
        if args.nt_dataset == 'gaussian':
            data = torch.randn(data.size()) + 0.5
            data = torch.clamp(data, 0, 1)
        if args.nt_dataset == 'uniform':
            data = torch.rand(data.size())

        noise = torch.randn(data.size()) + 0.5
        noise = torch.clamp(noise, 0, 1)
        data = (1-args.eps)*data + args.eps*noise

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.softmax(model(data))
        pred = output.data.max(1)[1] # get the index of the max log-probability
        equal_flag = pred.eq(target.data).cpu()
        correct += equal_flag.sum()
        for i in range(data.size(0)):
            count_over += 1
            prediction = (output.data[i]).cpu().numpy()
            entropy = -np.sum(np.multiply(prediction, np.log(np.add(prediction,0.00000001))))
            total_over_entropy += entropy
            #ent_log_nt_over.write(str(target.data[i])+','+str(entropy)+'\n')

    total_over_entropy /= count_over
    #ent_log_nt_over.write('\n Over entropy: {:.4f}, Final Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    total_over_entropy, correct, len(nt_test_loader.dataset),
    #    100. * correct / len(nt_test_loader.dataset)))
    print('\n Over entropy: {:.4f}, Final Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_over_entropy, correct, len(nt_test_loader.dataset),
        100. * correct / len(nt_test_loader.dataset)))


# measure the entropy
target_entropy_test()
non_target_entropy_test()
mnist_flag_in = 0
mnist_flag_out = 0
if args.dataset == 'mnist':
    mnist_flag_in = 1
if args.nt_dataset == 'mnist':
    mnist_flag_out = 1
elif args.nt_dataset == 'gaussian':
    mnist_flag_out = 2
elif args.nt_dataset == 'uniform':
    mnist_flag_out = 3
genlog.testData(model, test_loader, nt_test_loader, args.noi, args.temper, args.outf, mnist_flag_in, mnist_flag_out, args.imageSize, args.eps)
callog.metric("temp","Imagenet",args.outf)
#ent_callog.metric("temp","Imagenet",args.outf)
#true_callog.metric("temp","Imagenet",args.outf)
