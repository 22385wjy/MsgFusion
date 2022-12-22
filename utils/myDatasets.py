# coding: utf-8
import os
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]


class ImagePair(data.Dataset):

    def __init__(self, impath1, impath2, mode='RGB', transform=None):
        # print('begin with ImagePair __init__')
        self.impath1 = impath1
        self.impath2 = impath2
        self.mode = mode
        self.transform = transform

    def loader2(self, path):
        Im = Image.open(path)
        I_L = Im.convert('RGB')
        return I_L

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_pair(self):
        # global i1,i2
        if self.is_image_file(self.impath1):
            i1 = self.loader2(self.impath1)

        if self.is_image_file(self.impath2):
            i2 = self.loader2(self.impath2)

        if self.transform is not None:
            img1 = self.transform(i1)
            img2 = self.transform(i2)

        return img1, img2

    def get_source(self):
        if self.is_image_file(self.impath1):
            img1 = self.loader(self.impath1)
        if self.is_image_file(self.impath2):
            img2 = self.loader(self.impath2)
        return img1, img2


class Im_transform(data.Dataset):
    def __init__(self, im1, im2, mode='RGB', transform=None):
        # print('begin with ImagePair __init__')
        self.im1 = im1
        self.im2 = im2
        self.mode = mode
        self.transform = transform

    def get_pair(self):
        # global i1,i2
        if self.transform is not None:
            ig1 = self.im1.convert('RGB')
            img_1 = self.transform(ig1)

            ig2 = self.im2.convert('RGB')
            img_2 = self.transform(ig2)
        return img_1, img_2


class ImageSequence(data.Dataset):
    def __init__(self, is_folder=False, mode='RGB', transform=None, *impaths):
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.impaths = impaths

    def loader(self, path):
        return Image.open(path).convert(self.mode)

    def get_imseq(self):
        if self.is_folder:
            folder_path = self.impaths[0]
            impaths = self.make_dataset(folder_path)
        else:
            impaths = self.impaths

        imseq = []
        for impath in impaths:
            if os.path.exists(impath):
                im = self.loader(impath)
                if self.transform is not None:
                    im = self.transform(im)
                imseq.append(im)
        return imseq

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def make_dataset(self, img_root):
        images = []
        for root, _, fnames in sorted(os.walk(img_root)):
            for fname in fnames:
                if self.is_image_file(fname):
                    img_path = os.path.join(img_root, fname)
                    images.append(img_path)
        return images
