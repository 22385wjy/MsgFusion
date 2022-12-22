import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
from pytorch_msssim import ssim
import os
import os.path
import scipy.io as scio
from PIL import Image
import cv2
from tensorboardX import SummaryWriter
from matplotlib.font_manager import FontProperties
import torch.fft as fft
from skimage import color

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def load_onemodal():
    # load the train mri data  './data_add_t2pet/MR-T2/'
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/MR-T2/')  # './data_add_t2pet/mr168/'
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    # print(len(data))
    train_one = np.zeros((len(data), image_width, image_length, 3), dtype=float)  # (272,256,256,3)
    data_1 = np.zeros((len(data), image_width, image_length, 3))
    for i in range(len(data)):
        Im = Image.open(data[i])  # (256,256)
        data_1[i, :, :, :] = Im.convert('RGB')  # (256,256,3)
        # print(data[i].size)
        train_one[i, :, :, :] = (data_1[i, :, :, :] - np.min(data_1[i, :, :, :])) / (
                np.max(data_1[i, :, :, :]) - np.min(data_1[i, :, :, :]))
        train_one[i, :, :, :] = np.float32(train_one[i, :, :, :])
    # print(train_one.shape)
    train_one = train_one.transpose([0, 3, 1, 2])
    # print(train_one.shape)
    train_one_tensor = torch.from_numpy(train_one).float()
    print('load_onemodal ', train_one_tensor.shape)

    return train_one_tensor


def second_onemodal_mp():
    # load the train pet data  './data_add_t2pet/FDG/'
    dataset = os.path.join(os.getcwd(), '../data_add_t2pet/FDG/')  # './data_add_t2pet/pet168/'
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    train_other0 = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet0 = np.zeros((len(data), image_width, image_length), dtype=float)
    train_other = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet = np.zeros((len(data), image_width, image_length), dtype=float)
    train_one = np.zeros((len(data), image_width, image_length, 3), dtype=float)  # (272,256,256,3)
    train_v = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    train_v1 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    train_v2 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    data_0 = np.zeros((len(data), image_width, image_length, 3))
    data_2 = np.zeros((len(data), image_width, image_length, 3))
    data_hsv = np.zeros((len(data), image_width, image_length, 3))
    for i in range(len(data)):
        # *****************************to obtain v of hsv in pet********************************
        # # first way to get hsv
        train0 = Image.open(data[i])
        imRGBImage = train0.convert("RGB")  # 4 channel to 3 channel/RGBA-->RGB
        img_hsv = cv2.cvtColor(np.array(imRGBImage), cv2.COLOR_BGR2HSV)
        hsvChannels = cv2.split(img_hsv)
        v1 = hsvChannels[2]

        # # second way to get hsv
        train_other0[i, :, :, :] = (imageio.imread(data[i]))
        train_pet0[i, :, :] = train_other0[i, :, :, 0] + train_other0[i, :, :, 1] + train_other0[i, :, :, 2]

        Lightness1 = train_pet0[i, :, :] - v1

        data_0[i, :, :, :] = Image.fromarray(Lightness1).convert('RGB')
        train_v[i, :, :, :] = (data_0[i, :, :, :] - np.min(data_0[i, :, :, :])) / (
                    np.max(data_0[i, :, :, :]) - np.min(data_0[i, :, :, :]))
        train_v1[i, :, :, :] = np.float32(train_v[i, :, :, :])

        # *****************************to obtain gray image tensor of pet********************************
        train_other[i, :, :, :] = (imageio.imread(data[i]))
        train_pet[i, :, :] = 0.2989 * train_other[i, :, :, 0] + 0.5870 * train_other[i, :, :, 1] + 0.1140 * train_other[
                                                                                                            i, :, :, 2]
        Im = Image.fromarray(train_pet[i, :, :])  # numpy to Image
        data_2[i, :, :, :] = Im.convert('RGB')  # (256,256,3)
        train_one[i, :, :, :] = (data_2[i, :, :, :] - np.min(data_2[i, :, :, :])) / (
                np.max(data_2[i, :, :, :]) - np.min(data_2[i, :, :, :]))
        train_one[i, :, :, :] = np.float32(train_one[i, :, :, :])

    train_one = train_one.transpose([0, 3, 1, 2])
    # print(train_one.shape)
    train_v1 = train_v1.transpose([0, 3, 1, 2])

    train_one_tensor = torch.from_numpy(train_one).float()
    print('second_onemodal pet ', train_one_tensor.shape)
    train_v1_tensor = torch.from_numpy(train_v1).float()
    # print('second_onemodal train_v ',train_v1_tensor.shape)

    return train_one_tensor, train_v1_tensor


def second_onemodal_cm():
    # load the train pet data  './data_add_t2pet/FDG/'
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/FDG/')  # './data_add_t2pet/pet168/'
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    train_other0 = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet0 = np.zeros((len(data), image_width, image_length), dtype=float)
    train_other = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet = np.zeros((len(data), image_width, image_length), dtype=float)
    train_one = np.zeros((len(data), image_width, image_length, 3), dtype=float)  # (272,256,256,3)
    train_v = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    # train_v1 = np.zeros((len(data), image_width, image_length,3), dtype=float)
    train_v2 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    data_0 = np.zeros((len(data), image_width, image_length, 3))
    data_2 = np.zeros((len(data), image_width, image_length, 3))
    data_hsv = np.zeros((len(data), image_width, image_length, 3))
    for i in range(len(data)):

        # # to get hsv
        train_other0[i, :, :, :] = (imageio.imread(data[i]))
        train_pet0[i, :, :] = train_other0[i, :, :, 0] + train_other0[i, :, :, 1] + train_other0[i, :, :, 2]
        data_0[i, :, :, :] = Image.fromarray(train_pet0[i, :, :]).convert('RGB')
        data_hsv[i, :, :, :] = color.rgb2hsv(data_0[i, :, :, :])
        v2 = data_hsv[i, :, :, 2]

        Lightness2 = train_pet0[i, :, :] - v2

        data_0[i, :, :, :] = Image.fromarray(Lightness2).convert('RGB')
        train_v[i, :, :, :] = (data_0[i, :, :, :] - np.min(data_0[i, :, :, :])) / (
                    np.max(data_0[i, :, :, :]) - np.min(data_0[i, :, :, :]))
        train_v2[i, :, :, :] = np.float32(train_v[i, :, :, :])

        train_other[i, :, :, :] = (imageio.imread(data[i]))
        train_pet[i, :, :] = 0.2989 * train_other[i, :, :, 0] + 0.5870 * train_other[i, :, :, 1] + 0.1140 * train_other[
                                                                                                            i, :, :, 2]
        Im = Image.fromarray(train_pet[i, :, :])  # numpy to Image
        data_2[i, :, :, :] = Im.convert('RGB')  # (256,256,3)
        # print(data_2.shape)
        train_one[i, :, :, :] = (data_2[i, :, :, :] - np.min(data_2[i, :, :, :])) / (
                np.max(data_2[i, :, :, :]) - np.min(data_2[i, :, :, :]))
        train_one[i, :, :, :] = np.float32(train_one[i, :, :, :])
    # print(train_one.shape)
    # print(train_v.shape)
    train_one = train_one.transpose([0, 3, 1, 2])
    # print(train_one.shape)
    # train_v1 = train_v1.transpose([0, 3, 1, 2])
    train_v2 = train_v2.transpose([0, 3, 1, 2])
    # print(train_v.shape)
    train_one_tensor = torch.from_numpy(train_one).float()
    print('second_onemodal pet ', train_one_tensor.shape)
    # train_v1_tensor = torch.from_numpy(train_v1).float()
    train_v2_tensor = torch.from_numpy(train_v2).float()
    print('second_onemodal train_v ', train_v2_tensor.shape)

    return train_one_tensor, train_v2_tensor


class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

        # self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=3, stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()  # [1, 3, 256, 256]
        # (batch, c, h, w/2+1, 2)
        ffted = torch.rfft(x, signal_ndim=2, normalized=True)  # [1, 3, 256, 129, 2]
        print(ffted.shape)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [1, 3, 2, 256, 129]
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # [1, 6, 256, 129]
        print(ffted.shape)
        ffted = F.pad(ffted, (1, 1, 1, 1), 'replicate')
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1) #[1, 64, 256, 129]
        print(ffted.shape)
        ffted = self.relu(self.bn(ffted))  # [1, 64, 256, 129]
        print(ffted.shape)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2) #[1, 32, 256, 129, 2]
        print(ffted.shape)
        output = torch.irfft(ffted, signal_ndim=2,
                             signal_sizes=r_size[2:], normalized=True)  # [1, 32, 256, 256]
        print(output.shape)
        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=3, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, 3, kernel_size=3, bias=False)

    def forward(self, x):
        # print(x.shape)
        x = self.downsample(x)
        # print(x.shape)

        x = F.pad(x, (1, 1, 1, 1), 'replicate')
        x = self.conv1(x)
        # print(x.shape)

        ftensor1 = fft.fftn(x, dim=(2, 3))
        # print(ftensor1.shape)
        h = ftensor1.shape[2]
        w = ftensor1.shape[3]
        ftensor1 = torch.roll(ftensor1, shifts=(h // 2, w // 2), dims=(2, 3))
        tensor1_f = torch.abs(fft.ifftn(ftensor1, dim=(2, 3)))
        # print(tensor1_f.shape)
       
        output = tensor1_f
        return output


class FourierUnit0(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit0, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()  # [1, 3, 256, 256]
        # (batch, c, h, w/2+1, 2)
        ffted = torch.rfft(x, signal_ndim=2, normalized=True)  # [1, 3, 256, 129, 2]
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [1, 3, 2, 256, 129]
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])  # [1, 6, 256, 129]
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1) #[1, 64, 256, 129]
        ffted = self.relu(self.bn(ffted))  # [1, 64, 256, 129]

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2) #[1, 32, 256, 129, 2]

        output = torch.irfft(ffted, signal_ndim=2,
                             signal_sizes=r_size[2:], normalized=True)  # [1, 32, 256, 256]

        return output


class MsgBIF1(nn.Module):
    def __init__(self, resnet):
        super(MsgBIF1, self).__init__()
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 64)
        self.conv7 = ConvBlock(256, 256)
        self.conv8 = ConvBlock(256, 128)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = ConvBlock(192, 128)
        self.conv11 = ConvBlock(384, 256)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.convfcout = nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1, bias=True)

        self.SpT = SpectralTransform(in_channels=3, out_channels=128)

        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

        self.pool = nn.MaxPool2d(2, 2)

        # pet lf layer
        self.pet_lf = nn.Sequential(  # input shape (,3,256,256)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))  # output shape (,64,256,256)
        # pet hf layer
        self.pet_hf = nn.Sequential(  # input shape (,3,256,256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_min(self, tensors):
        min_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                min_tensor = tensor
            else:
                min_tensor = torch.min(min_tensor, tensor)
        return min_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def operate(self, operator, tensors):
        out_tensors = operator(tensors)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = F.pad(tensors, padding, mode=mode, value=value)

        return out_tensors

    def spatial_work(self, tensor1, tensor2):
        # Feature extraction
        shape = tensor2.size()
        # spatial1 = tensor1.sum(dim=1, keepdim=True)
        spatial1 = tensor1.mean(dim=1, keepdim=True)
        spatial2 = tensor2.mean(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def channel_work(self, tensors):
        # print('channel_work')
        # print('tensors: ',tensors.shape)
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3),
                                   mode='replicate')
        # print(outs.shape)
        x1c1 = self.conv1(outs)
        # print(x1c1.shape)
        x1p1 = self.pool(x1c1)
        # print(x1p1.shape)
        x1c2 = self.conv2(x1p1)
        # print(x1c2.shape)
        x1c3 = self.conv3(x1c2)
        # print(x1c3.shape)
        self.up = nn.Upsample(scale_factor=2)
        x1up = self.up(x1c3)
        # print(x1up.shape)

        x1pup = self.up(x1p1)
        # print(x1pup.shape)
        x1c2up = self.up(x1c2)
        # print(x1c2up.shape)
        cat1 = torch.cat((x1up, x1pup, x1c2up), 1)
        # print(cat1.shape)
        x1c7 = self.conv7(cat1)
        # print(x1c7.shape)

        x1c12 = self.conv2(x1c1)
        x1c123 = self.conv3(x1c12)
        x1c1234 = self.conv4(x1c123)
        # print(x1c1234.shape)

        out1 = self.tensor_max([x1c7, x1c1234])
        # print(out1.shape)
        outx1 = self.conv5(out1)
        # print(outx1.shape)#64

        x1c8 = self.conv8(x1c7)
        # print(x1c8.shape)
        x1c9 = self.conv9(x1c8)
        # print(x1c9.shape)
        cat2 = torch.cat((x1c9, outx1), 1)
        # print(cat2.shape)
        x1catc9 = self.conv9(cat2)
        # print(x1catc9.shape)#64

        catx1 = torch.cat((x1catc9, outx1), 1)
        # print(catx1.shape)
        x1cat = self.conv9(catx1)
        # print(x1cat.shape)

        ######
        x2c2 = self.conv2(x1c1)  # 1,64,256,256
        # print(x2c2.shape)

        cat12 = torch.cat((x1c1, x2c2), 1)
        # print(cat12.shape)
        x2catc9 = self.conv9(cat12)  # 128->64
        # print(x2catc9.shape)
        x2catc3 = self.conv3(x2catc9)
        # print(x2catc3.shape)

        cat23 = torch.cat((x2c2, x2catc3), 1)
        # print(cat23.shape)
        x2catc10 = self.conv10(cat23)  # 192->128
        # print(x2catc10.shape)
        x2catc4 = self.conv4(x2catc10)
        # print(x2catc4.shape)

        cat34 = torch.cat((x2catc3, x2catc4), 1)
        # print(cat34.shape)
        x2catc11 = self.conv11(cat34)
        # print(x2catc11.shape)
        x2cat = self.conv5(x2catc11)
        # print(x2cat.shape)

        ##final connection
        cat = torch.cat((x2cat, x1cat), 1)
        # print(cat.shape)
        xcatc9 = self.conv9(cat)
        # print(xcatc9.shape)
        out = self.conv2(xcatc9)
        # print(out.shape)

        out = torch.tanh(out)

        return out

    def fusion2(self, *tensors):
        # Feature fusion
        out = self.tensor_max(tensors)
        out = self.conv6(out)
        out = torch.tanh(out)
        return out

    def fusion1(self, *tensors):
        # Feature fusion
        out = self.tensor_max(tensors)
        # Feature reconstruction
        out = self.conv2(out)
        return out

    def twoImageFuse(self, t1, t2, light):
        tensor1 = t2
        tensor2 = t1
        # _______________________________________________
        cb1 = self.channel_work(tensor1)
        # _______________________________________________
        sb = self.pet_lf(tensor2)
        fu2 = self.SpT(tensor2)
        fs = self.spatial_work(fu2, sb)
        # _______________________________________________
        cbfs1 = self.tensor_max([cb1, fs])
        out = self.conv6(cbfs1)
        cbfs12 = torch.tanh(out)
        light2lf = self.pet_lf(light)
        out = self.conv6(light2lf)
        cbfs2 = torch.tanh(out)
        cbfs2 = self.spatial_work(light, cbfs2)
        cbfs1l2 = self.tensor_max([cbfs12, cbfs2])

        return cbfs1l2

    def forward(self, tensor1, tensor2, light):

        res12 = self.twoImageFuse(tensor1, tensor2, light)
        return res12


def myNet1():
    resnet = models.resnet101(pretrained=True)
    mynet1 = MsgBIF1(resnet).to(device)
    # mynet=MsgBIF().to(device)
    mynet = mynet1.float()
    return mynet


class MsgBIF2(nn.Module):
    def __init__(self, resnet):
        super(MsgBIF2, self).__init__()
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 64)
        self.conv7 = ConvBlock(256, 256)
        self.conv8 = ConvBlock(256, 128)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = ConvBlock(192, 128)
        self.conv11 = ConvBlock(384, 256)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.convfcout = nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1, bias=True)

        self.FFC = FourierUnit0(in_channels=3, out_channels=32)
        self.SpT = SpectralTransform(in_channels=3, out_channels=128)

        # Initialize other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

        self.pool = nn.MaxPool2d(2, 2)

        # pet lf layer
        self.pet_lf = nn.Sequential(  # input shape (,3,256,256)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))  # output shape (,64,256,256)
        # pet hf layer
        self.pet_hf = nn.Sequential(  # input shape (,3,256,256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_min(self, tensors):
        min_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                min_tensor = tensor
            else:
                min_tensor = torch.min(min_tensor, tensor)
        return min_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def operate(self, operator, tensors):
        out_tensors = operator(tensors)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = F.pad(tensors, padding, mode=mode, value=value)

        return out_tensors

    def spatial_work(self, tensor1, tensor2):
        # Feature extraction
        shape = tensor2.size()
        # spatial1 = tensor1.sum(dim=1, keepdim=True)
        spatial1 = tensor1.mean(dim=1, keepdim=True)
        spatial2 = tensor2.mean(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

        return tensor_f

    def spatial_work_one(self, tensor1):
        print('spatial_work_one')
        # Feature extraction
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        print(tensor1.shape)
        tensor1 = self.operate(self.conv1, tensor1)
        print(tensor1.shape)
        shape = tensor1.size()
        # spatial1 = tensor1.sum(dim=1, keepdim=True)
        spatial1 = tensor1.mean(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1
        print(tensor_f.shape)

        img = (np.transpose(tensor_f[0].cpu().data.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.imshow(img[:, :, 0])
        plt.show()

        return tensor_f

    def channel_work(self, tensors):
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3),
                                   mode='replicate')
        # print(outs.shape)
        x1c1 = self.conv1(outs)
        # print(x1c1.shape)
        x1p1 = self.pool(x1c1)
        # print(x1p1.shape)
        x1c2 = self.conv2(x1p1)
        # print(x1c2.shape)
        x1c3 = self.conv3(x1c2)
        # print(x1c3.shape)
        self.up = nn.Upsample(scale_factor=2)
        x1up = self.up(x1c3)
        # print(x1up.shape)

        x1pup = self.up(x1p1)
        # print(x1pup.shape)
        x1c2up = self.up(x1c2)
        # print(x1c2up.shape)
        cat1 = torch.cat((x1up, x1pup, x1c2up), 1)
        # print(cat1.shape)
        x1c7 = self.conv7(cat1)
        # print(x1c7.shape)

        x1c12 = self.conv2(x1c1)
        x1c123 = self.conv3(x1c12)
        x1c1234 = self.conv4(x1c123)
        # print(x1c1234.shape)

        out1 = self.tensor_max([x1c7, x1c1234])
        # print(out1.shape)
        outx1 = self.conv5(out1)
        # print(outx1.shape)#64

        x1c8 = self.conv8(x1c7)
        # print(x1c8.shape)
        x1c9 = self.conv9(x1c8)
        # print(x1c9.shape)
        cat2 = torch.cat((x1c9, outx1), 1)
        # print(cat2.shape)
        x1catc9 = self.conv9(cat2)
        # print(x1catc9.shape)#64

        catx1 = torch.cat((x1catc9, outx1), 1)
        # print(catx1.shape)
        x1cat = self.conv9(catx1)
        # print(x1cat.shape)

        ######
        x2c2 = self.conv2(x1c1)  # 1,64,256,256
        # print(x2c2.shape)

        cat12 = torch.cat((x1c1, x2c2), 1)
        # print(cat12.shape)
        x2catc9 = self.conv9(cat12)  # 128->64
        # print(x2catc9.shape)
        x2catc3 = self.conv3(x2catc9)
        # print(x2catc3.shape)

        cat23 = torch.cat((x2c2, x2catc3), 1)
        # print(cat23.shape)
        x2catc10 = self.conv10(cat23)  # 192->128
        # print(x2catc10.shape)
        x2catc4 = self.conv4(x2catc10)
        # print(x2catc4.shape)

        cat34 = torch.cat((x2catc3, x2catc4), 1)
        # print(cat34.shape)
        x2catc11 = self.conv11(cat34)
        # print(x2catc11.shape)
        x2cat = self.conv5(x2catc11)
        # print(x2cat.shape)

        ##final connection
        cat = torch.cat((x2cat, x1cat), 1)
        # print(cat.shape)
        xcatc9 = self.conv9(cat)
        # print(xcatc9.shape)
        out = self.conv2(xcatc9)
        # print(out.shape)
        out = torch.tanh(out)

        return out

    def fusion2(self, *tensors):
        # Feature fusion
        out = self.tensor_max(tensors)
        out = self.conv6(out)
        out = torch.tanh(out)
        return out

    def fusion1(self, *tensors):
        # Feature fusion
        out = self.tensor_max(tensors)
        # Feature reconstruction
        out = self.conv2(out)
        return out

    def channel_to_one(self, tensor1):
        cb_1 = self.channel_work(tensor1)
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor1c1 = self.conv1(tensor1)
        cb_1 = self.tensor_max([cb_1, tensor1c1])

        res1 = self.conv2(cb_1)

        res1 = torch.tanh(res1)
        return res1

    def channel_to_one_old(self, tensor1, tensor2):
        cb_1 = self.channel_work(tensor1)
        cb_2 = self.channel_work(tensor2)
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor2 = self.tensor_padding(tensors=tensor2, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor1c1 = self.conv1(tensor1)
        tensor2c1 = self.conv1(tensor2)
        # print(tensor2c1.shape)
        cb_1 = self.tensor_max([cb_1, tensor1c1])
        cb_2 = self.tensor_max([cb_2, tensor2c1])
        # print(cb_2.shape)

        res1 = self.fusion1(cb_1, cb_2)
        img = (np.transpose(res1[0].cpu().data.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.imshow(img[:, :, 0])
        plt.show()
        return res1

    def fft_l_h(self, tensor1):
        h = tensor1.shape[2]
        w = tensor1.shape[3]
        lpf = torch.zeros((h, w))  # torch.zeros((h, w))
        R = (h + w) // 8  # or other
        for x in range(w):
            for y in range(h):
                if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                    lpf[y, x] = 1
        hpf = 1 - lpf
        hpf, lpf = hpf.to(device), lpf.to(device)

        # fft
        # f1 = fft.fftn(tensor1, dim=(2, 3))
        f1 = fft.fftn(tensor1)
        f1_l = f1 * lpf  
        f1_h = f1 * hpf  

        tensor1_l = torch.abs(fft.ifftn(f1_l))
        tensor1_h = torch.abs(fft.ifftn(f1_h))

        tensor1 = self.tensor_mean([tensor1_l, tensor1_h])
        tensor1_himg = (np.transpose(tensor1[0].cpu().data.numpy(), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.imshow(tensor1_himg[:, :, 0])
        plt.show()

        return tensor1_l, tensor1_h, tensor1

    def fft_(self, tensor1, tensor2):
        tensor1 = self.tensor_padding(tensors=tensor1, padding=(3, 3, 3, 3),
                                      mode='replicate')
        tensor2 = self.tensor_padding(tensors=tensor2, padding=(3, 3, 3, 3), mode='replicate')

        tensor1 = self.operate(self.conv1, tensor1)
        tensor2 = self.operate(self.conv1, tensor2)

        # fft and ifftn
        ftensor1 = fft.fftn(tensor1, dim=(2, 3))
        tensor1_f = torch.abs(fft.ifftn(ftensor1, dim=(2, 3)))
        ftensor2 = fft.fftn(tensor2, dim=(2, 3))
        tensor2_f = torch.abs(fft.ifftn(ftensor2, dim=(2, 3)))

        shape = tensor2.size()
        # spatial1 = tensor1.mean(dim=1, keepdim=True)
        spatial1 = tensor1_f.sum(dim=1, keepdim=True)
        spatial2 = tensor2_f.sum(dim=1, keepdim=True)
        # get weight map, soft-max
        EPSILON = 1e-5
        spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
        spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
        spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
        tensor_f = spatial_w1 * tensor1_f + spatial_w2 * tensor2_f

        return tensor_f

    def twoImageFuse(self, t1, t2, light):
        tensor1 = t2
        tensor2 = t1
        # _______________________________________________
        cb1 = self.channel_work(tensor1)
        # _______________________________________________
        sb = self.pet_lf(tensor2)
        fu2 = self.SpT(tensor2)
        fs = self.spatial_work(fu2, sb)
        # _______________________________________________
        cbfs1 = self.tensor_max([cb1, fs])
        out = self.conv6(cbfs1)
        cbfs12 = torch.tanh(out)
        light2lf = self.pet_lf(light)
        out = self.conv6(light2lf)
        cbfs2 = torch.tanh(out)
        cbfs2 = self.spatial_work(light, cbfs2)
        cbfs1l2 = self.tensor_max([cbfs12, cbfs2])

        return cbfs1l2

    def forward(self, tensor1, tensor2, light2):

        res12 = self.twoImageFuse(tensor1, tensor2, light2)
        return res12


def myNet2():
    resnet = models.resnet101(pretrained=True)
    mynet2 = MsgBIF2(resnet).to(device)
    mynet = mynet2.float()
    return mynet


def train_for_newModel(net, first_im, second_im, light, OutFilePath, ParameterNum):
    writer = SummaryWriter(OutFilePath + '/outLog/add_scalar_log_' + ParameterNum)
    save_file = open(OutFilePath + '/outTxt/Log_' + ParameterNum + '.txt', 'a')
    save_file.write(
        'Epoch' + ' ' + 'SSIM_MRI_Loss' + ' ' + 'SSIM_PET_Loss' + ' ' + 'L2_MRI_Loss' + ' ' + 'L2_PET_Loss' + ' ' + 'Total_Loss' + '\n')
    save_file2 = open(OutFilePath + '/outTxt/Log_TotalLoss_' + ParameterNum + '.txt', 'a')
    save_file2.write('Epoch' + 'Total_Loss' + '\n')

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all dtn parameters
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  
    l2_loss = nn.MSELoss()  # MSEloss
    # perform the training
    counter = 0
    lamda = 0.8
    gamma_ssim = 0.5
    gamma_l2 = 0.5
    ep_ssim_mri_loss = []
    ep_ssim_pet_loss = []
    ep_l2_mri_loss = []
    ep_l2_pet_loss = []
    ep_loss = []
    loss_history = []
    for epoch in range(EPOCH):
        ssim_mri_Loss = []
        ssim_pet_Loss = []
        l2_mri_Loss = []
        l2_pet_Loss = []
        # run batch images
        batch_idxs = 555 // batch_size
        for idx in range(0, batch_idxs):
            b_x = first_im[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            b_y = second_im[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            # print('idx= ', idx,'b_x.shape', b_x.shape,'---b_y.shape ',b_y.shape)
            l2 = light[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            counter += 1
            output = net(b_x, b_y, l2)  # output
            ssim_loss_mri = 1 - ssim(output, b_x, data_range=1)
            ssim_loss_pet = 1 - ssim(output, b_y, data_range=1)
            l2_loss_mri = l2_loss(output, b_x)
            l2_loss_pet = l2_loss(output, b_y)
            ssim_total = gamma_ssim * ssim_loss_mri + (1 - gamma_ssim) * ssim_loss_pet
            l2_total = gamma_l2 * l2_loss_mri + (1 - gamma_l2) * l2_loss_pet
            loss_total = lamda * ssim_total + (1 - lamda) * l2_total
            optimizer.zero_grad()  # clear gradients for this training step
            loss_total.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # store all the loss values at each epoch
            ssim_mri_Loss.append(ssim_loss_mri.item())
            ssim_pet_Loss.append(ssim_loss_pet.item())
            l2_mri_Loss.append(l2_loss_mri.item())
            l2_pet_Loss.append(l2_loss_pet.item())

            loss_history.append(loss_total.item())

            save_file2.write(
                str(epoch) + ' ' + str(loss_history[-1]) + '\n')

            if counter % 15 == 0:
                print(
                    "Epoch: [%2d],step: [%2d], mri_ssim_loss: [%.8f], pet_ssim_loss: [%.8f],  total_ssim_loss: [%.8f], total_l2_loss: [%.8f], total_loss: [%.8f]"
                    % (epoch, counter, ssim_loss_mri, ssim_loss_pet, ssim_total, l2_total, loss_total))

        scheduler.step()  # updata learning_rate

        av_ssim_mri_loss = np.average(ssim_mri_Loss)
        ep_ssim_mri_loss.append(av_ssim_mri_loss)

        av_ssim_pet_loss = np.average(ssim_pet_Loss)
        ep_ssim_pet_loss.append(av_ssim_pet_loss)

        av_l2_mri_loss = np.average(l2_mri_Loss)
        ep_l2_mri_loss.append(av_l2_mri_loss)

        av_l2_pet_loss = np.average(l2_pet_Loss)
        ep_l2_pet_loss.append(av_l2_pet_loss)

        av_loss = np.average(loss_history)
        ep_loss.append(av_loss)

        writer.add_scalars("Epoch/Loss", {'ssim_mri_Loss': ep_ssim_mri_loss[-1], 'ssim_pet_Loss': ep_ssim_pet_loss[-1],
                                          'l2_mri_Loss': ep_l2_mri_loss[-1], 'l2_pet_Loss': ep_l2_pet_loss[-1],
                                          'loss_history': ep_loss[-1]}, epoch)

        save_file.write(
            str(epoch) + ' ' + str(ep_ssim_mri_loss[-1]) + ' ' + str(ep_ssim_pet_loss[-1]) + ' ' + str(
                ep_l2_mri_loss[-1]) + ' ' + str(
                ep_l2_pet_loss[-1]) + ' ' + str(ep_loss[-1]) + '\n')

        if (epoch == EPOCH - 1):
            # Save a checkpoint
            torch.save(net.state_dict(), OutFilePath + '/outModel/MsgBIF_' + ParameterNum + '.pth')

    writer.close()
    l1 = np.asarray(ep_ssim_mri_loss)
    l2 = np.asarray(ep_ssim_pet_loss)
    l3 = np.asarray(ep_l2_mri_loss)
    l4 = np.asarray(ep_l2_pet_loss)
    l5 = np.asarray(ep_loss)

    fontP = FontProperties()
    fontP.set_size('large')
    plt.plot(l1, 'b', label='$SSIM_{MRI}$')
    plt.plot(l2, 'c', label='$SSIM_{PET}$')
    plt.plot(l3, 'g', label='$L2_{MRI}$')
    plt.plot(l4, 'y', label='$L2_{PET}$')
    plt.plot(l5, 'r', label='$Loss$')
    fig = plt.gcf()
    fig.savefig(OutFilePath + '/outCurve/loss_SSIMorL2_MRorPET_' + ParameterNum + '.png')

    return loss_history


if __name__ == "__main__":
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    OutFilePath = './Out21Ft2_4'
    ParameterNum = '0.8'
    print(device)
    image_length = 256
    image_width = 256
    mr_channels = 1
    gray_channels = 1
    pet_channels = 4
    rgb_channels = 2
    batch_size = 2
    EPOCH = 251
    learning_rate = 0.001
    mean = [0, 0, 0]  # normalization parameters
    std = [1, 1, 1]

    DataSetType1 = 'mr_pet'
    DataSetType2 = 'ct_mr'
    DataSetType = DataSetType2

    first_one = load_onemodal()
    if DataSetType == DataSetType1:
        second_one, light = second_onemodal_mp()
        ourmodel = myNet1()  # Our model (MsgBIF)
    else:
        second_one, light = second_onemodal_cm()
        ourmodel = myNet2()  # Our model (MsgBIF)

    loss_history = train_for_newModel(ourmodel, first_one, second_one, light, OutFilePath, ParameterNum)

    print('finish fusing')
