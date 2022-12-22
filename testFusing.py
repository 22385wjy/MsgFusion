import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
import os
import os.path
from PIL import Image
import matplotlib
import scipy.misc
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from utils.myTransforms import denorm, norms, detransformcv2
from utils.myDatasets import ImagePair, Im_transform
import cv2
from torchvision.utils import save_image
import time
from skimage import color
from MsgFusion_train import \
    myNet2  # 1) you need choose one model in terms of different modal fusion (myNet1:MR-PET, myNet2:CT-MR)


def to_transform0(pathct, pathmr):
    pair_loader0 = ImagePair(impath1=pathct, impath2=pathmr,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)
                             ]))
    imgct, imgmr = pair_loader0.get_pair()

    imgct.unsqueeze_(0)
    imgmr.unsqueeze_(0)
    imgct = Variable(imgct.cpu(), requires_grad=False)
    imgmr = Variable(imgmr.cpu(), requires_grad=False)

    return imgct, imgmr


def getVinHSV(petim):
    # data=petim
    dataset = os.path.join(os.getcwd(), petim)  # './data_add_t2pet/pet168/'
    # data = glob.glob(os.path.join(dataset, "*.*"))
    data = glob.glob(os.path.join(dataset))
    data = natsort.natsorted(data, reverse=False)
    train_other0 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    train_pet0 = np.zeros((len(data), image_width, image_length), dtype=float)
    train_v = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    train_v1 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    train_v2 = np.zeros((len(data), image_width, image_length, 3), dtype=float)
    data_0 = np.zeros((len(data), image_width, image_length, 3))
    data_hsv = np.zeros((len(data), image_width, image_length, 3))

    for i in range(len(data)):
        # *****************************to obtain v of hsv ********************************
        # # first way to get hsv
        # print(data[i])
        train0 = Image.open(data[i])
        imRGBImage = train0.convert("RGB")  # 4 channel to 3 channel/RGBA-->RGB
        img_hsv = cv2.cvtColor(np.array(imRGBImage), cv2.COLOR_BGR2HSV)
        hsvChannels = cv2.split(img_hsv)
        v1 = hsvChannels[2]

        # # second way to get hsv
        # print((imageio.imread(data[i])).shape)
        train_other0[i, :, :, :] = (imageio.imread(data[i]))
        train_pet0[i, :, :] = train_other0[i, :, :, 0] + train_other0[i, :, :, 1] + train_other0[i, :, :, 2]
        data_0[i, :, :, :] = Image.fromarray(train_pet0[i, :, :]).convert('RGB')
        data_hsv[i, :, :, :] = color.rgb2hsv(data_0[i, :, :, :])
        v2 = data_hsv[i, :, :, 2]

        Lightness1 = train_pet0[i, :, :] - v1
        Lightness2 = train_pet0[i, :, :] - v2

        data_0[i, :, :, :] = Image.fromarray(Lightness1).convert('RGB')
        train_v[i, :, :, :] = (data_0[i, :, :, :] - np.min(data_0[i, :, :, :])) / (
                np.max(data_0[i, :, :, :]) - np.min(data_0[i, :, :, :]))
        train_v1[i, :, :, :] = np.float32(train_v[i, :, :, :])
        data_0[i, :, :, :] = Image.fromarray(Lightness2).convert('RGB')
        train_v[i, :, :, :] = (data_0[i, :, :, :] - np.min(data_0[i, :, :, :])) / (
                np.max(data_0[i, :, :, :]) - np.min(data_0[i, :, :, :]))
        train_v2[i, :, :, :] = np.float32(train_v[i, :, :, :])

    train_v1 = train_v1.transpose([0, 3, 1, 2])
    train_v2 = train_v2.transpose([0, 3, 1, 2])
    # print(train_v2.shape)
    light1 = torch.from_numpy(train_v1).float()
    light2 = torch.from_numpy(train_v2).float()
    # print(light2.shape)
    return light1, light2


def to_transform(pathct, pathmr):
    light1, light2 = getVinHSV(pathmr)
    pair_loader0 = ImagePair(impath1=pathct, impath2=pathmr,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)
                             ]))
    imgct, imgmr = pair_loader0.get_pair()

    imgct.unsqueeze_(0)
    imgmr.unsqueeze_(0)
    imgct = Variable(imgct.cpu(), requires_grad=False)
    imgmr = Variable(imgmr.cpu(), requires_grad=False)

    return imgct, imgmr, light1, light2


if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    image_length = 256
    image_width = 256
    mean = [0, 0, 0]
    std = [1, 1, 1]
    DataSetType1 = 'mr_pet'
    DataSetType2 = 'ct_mr'
    DataSetType = DataSetType2  # 2) you need choose one combination to do fusion
    modelName = str(DataSetType)

    # root_path1 = './testData/BrainTumor2/MR/'  # 3) you need to change dataset path
    # root_path2 = './testData/BrainTumor2/PET/'
    # save_path = './Fused_Images/MR_PET/'

    root_path1 = './testData/Fatal stroke/CT/'
    root_path2 = './testData/Fatal stroke/MR/'
    save_path = './Fused_Images/CT_MR/'

    # load the model
    model = myNet()  # 4) the same as import myNet1 or myNet2
    model_path = './Models_Fourier/' + DataSetType + '/outModel/' + DataSetType + '.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    file1 = []
    file2 = []
    for root, dir, files1 in os.walk(root_path1):
        for file in files1:
            pathct = root_path1 + str(file)
            file1.append(pathct)
    for root, dir, files2 in os.walk(root_path2):
        for file in files2:
            pathmr = root_path2 + str(file)
            file2.append(pathmr)

    print(" ---> wait several minutes........")
    for i in range(len(file1)):
        img1, img2, light1, light2 = to_transform(file1[i], file2[i])
        # name2=str(i)
        if DataSetType == DataSetType1:
            f1 = model.twoImageFuse(img1, img2, light1)
        else:
            f1 = model.twoImageFuse(img1, img2, light2)

        pre = file1[i].split('/')[4]
        last = pre.split('.jpg')[0]
        name1 = str(last)
        print('image', i, ' name: ', name1)

        # -- to save tensor image ---
        res = denorm(mean, std, f1[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = res_img.transpose([1, 2, 0])

        i2_path = save_path + 'Gray/'
        if not os.path.exists(i2_path):
            os.makedirs(i2_path)
        i3_path = save_path + 'Color/'
        if not os.path.exists(i3_path):
            os.makedirs(i3_path)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imageio.imsave(i2_path + name1 + '.png', img)  # gray image
        matplotlib.image.imsave(i3_path + name1 + '.png', img)  # purple background

    print(" Alright, all test images pairs fused........")
