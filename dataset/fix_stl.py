import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision import datasets
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

from .randaugment import RandAugmentMC

# Parameters for data
stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

# Augmentations.
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std)
])

transform_strong = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(stl10_mean, stl10_std)
])

def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])
class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3


def get_stl(root,args):
    train_labeled_dataset = datasets.STL10(root, split="train", transform=transform_train, download=True)
    train_unlabeled_dataset = datasets.STL10(root, split="unlabeled",
                                                         transform=TransformFixMatchSTL(mean=stl10_mean, std=stl10_std),
                                                         download=True)
    test_dataset = datasets.STL10(root, split="test", transform=transform_val, download=True)


    l_samples = make_imb_data(args.num_max, 10, args.imb_ratio, 1, 0)
    train_labeled_idxs = train_split_l(train_labeled_dataset.labels, l_samples, args)
    train_labeled_dataset = make_imbalance(train_labeled_dataset, train_labeled_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def testsplit(labels):
    labels = np.array(labels)
    test_idxs=[]
    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        test_idxs.extend(idxs[:1500])
    np.random.shuffle(test_idxs)
    return test_idxs
def train_split(labels, n_labeled_per_class, n_unlabeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs




def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    print(class_num_list)
    return list(class_num_list)

def train_split_l(labels, n_labeled_per_class, args):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs

def make_imbalance(dataset, indexs):
    dataset.data = dataset.data[indexs]
    dataset.labels = dataset.labels[indexs]
    return dataset


class TransformFixMatchSTL(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)