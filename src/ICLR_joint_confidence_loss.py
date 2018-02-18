###########################################################
# OutGAN code based on sample code from pytorch           #
# Writer: Kimin Lee 					                  #
###########################################################
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unnorm_data_loader as data_loader
import numpy as np
import model as CNN
import models_from_vision as models

import torchvision.utils as vutils
from torch.utils.serialization import load_lua
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='OutGAN')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--dataset', required=True, help='target dataset: cifar10 | mnist | imagenet | SVHN')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--decreasing_lr', default='60,120,160', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes (default: 10)')
parser.add_argument('--alpha', type=float, default=1, help='penalty parameter for KL divergence')
parser.add_argument('--Classification_Net', default='VGG13', help="Type of Classification Nets")
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--adamlr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--droprate', type=float, default=0.2, help='learning rate decay')
#parser.add_argument('--pre_trained_net', required=True, help="path to pre trained_net")
#parser.add_argument('--pre_trained_gan', required=True, help="path to pre-trained GAN")
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--optimizer_flag', default='adam', help="Type of optimizer")
parser.add_argument('--pre_epoch', type=int, default=200)
parser.add_argument('--noise_flag', type=int, default=0, help="Type of optimizer")
parser.add_argument('--gamma', type=float, default=0.1, help='gamma for adam. default=0.5')


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

print('load classification network: ',args.Classification_Net)
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

if args.Classification_Net == 'L2':
    model = CNN.CNN_2(num_c=args.num_classes)
    model.apply(weights_init)
elif args.Classification_Net == 'ALEX':
    model = models.alexnet()
elif args.Classification_Net == 'VGG13':
    model = models.vgg13()
#model.load_state_dict(torch.load(args.pre_trained_net))
print(model)

print('load GAN')
ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 3
netG = CNN.BadGenerator(ngpu, nz, ngf, nc)
netG.apply(weights_init)
#netG.load_state_dict(torch.load('%s/netG_epoch_%s.pth'%(args.pre_trained_gan, args.pre_epoch)))
netD = CNN.Discriminator(ngpu, nc, ndf)
netD.apply(weights_init)
#netD.load_state_dict(torch.load('%s/netD_epoch_%s.pth'%(args.pre_trained_gan, args.pre_epoch)))

# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

if args.cuda:
    model.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

if args.optimizer_flag == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.optimizer_flag == 'adam':
    args.lr = 0.001
    if args.Classification_Net == 'VGG13':
        args.lr = 0.0002
    print(args.lr)
    args.wd = 0.00
    args.droprate = 0.1
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

optimizerD = optim.Adam(netD.parameters(), lr=args.adamlr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.adamlr, betas=(args.beta1, 0.999))

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1./args.num_classes))
        if args.cuda:
            data, target, gan_target, uniform_dist = data.cuda(), target.cuda(), gan_target.cuda(), uniform_dist.cuda()
        data, target, uniform_dist = Variable(data), Variable(target), Variable(uniform_dist)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterion(output, targetv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, targetv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) - H(G(x)) + log P_{in}(x) 1[P_{in}(x)>e]
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))  
        output = netD(fake)
        errG = criterion(output, targetv)
        D_G_z2 = output.data.mean()
        # pull-away term
        '''
        numer = fake.view(fake.size(0),fake.size(1)*fake.size(2)*fake.size(3))
        norm_fake = torch.matmul(numer,numer.transpose(0,1))
        denominator = torch.norm(numer,2,1,True)
        denorm = torch.matmul(denominator,denominator.transpose(0,1))
        temp_zero = torch.Tensor(denorm.size()).fill_(0).cuda()
        zero_mat = Variable(temp_zero)
        pt = torch.addcdiv(zero_mat, norm_fake, denorm)
        zero_mat_2 = Variable(temp_zero)
        pt = torch.addcmul(zero_mat_2, pt, Variable(torch.Tensor(data.size(0),data.size(0)).fill_(1).cuda()-torch.eye(data.size(0)).cuda()))
        pt = torch.addcmul(zero_mat_2, pt, pt)
        pt = torch.sum(pt)/(data.size(0)*(data.size(0)-1))
        '''
        pt = 0
        # minimize the true distribution
        KL_fake_output = F.log_softmax(model(fake))
        errG_KL = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes*args.alpha
        generator_loss = errG + errG_KL + pt
        generator_loss.backward()
        optimizerG.step()

        ############################
        # (3) Update classification network log P(y|x) + log P_{in}(x) 1[P_{in}(x)>e]
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target)
        # KL divergence
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        noise = Variable(noise)
        fake = netG(noise)
        KL_fake_output = F.log_softmax(model(fake))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*args.num_classes*args.alpha
        syn_loss = 0
        if args.noise_flag == 1:
            syn_data = torch.randn(fake.size()) + 0.5
            syn_data = torch.clamp(syn_data, 0, 1).cuda()
            syn_output = F.log_softmax(model(Variable(syn_data)))
            syn_loss += F.kl_div(syn_output, uniform_dist)*args.num_classes*args.alpha
        '''
        KL_regul = F.log_softmax(model(data))
        KL_regul_loss = F.kl_div(KL_regul, uniform_dist)*args.num_classes*args.gamma
        '''
        total_loss = loss + KL_loss_fake + syn_loss
        total_loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], 0))
            print('GAN Train Epoch: {} [{}/{}] GAN Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    if epoch % 20 == 0:
        vutils.save_image(data.data.cpu(), '%s/real_samples.png'%args.outf, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png'%(args.outf, epoch), normalize=True)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.dataset == 'mnist':
            data = data.expand(data.size(0), 3, args.imageSize, args.imageSize).clone()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    test(epoch)
    if epoch % 50 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))

