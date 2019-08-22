from torchvision import datasets, transforms
from base import BaseDataLoader

import numpy as np
import os, glob
import pandas as pd
import re
from pathlib import Path

#Data management
from torch.utils.data import Dataset
from torch.utils.data import random_split
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import torch.tensor

#Image processing
import cv2 as cv
from torchvision import transforms

import random

default_point_orders = {
    "bel_wings_1": [4,5,6,7,3,2,1,0],
    "bel_wings_2": [3,4,5,6,7,2,1,0],
    "bel_wings_3": [1,3,4,6,7,5,2,0],
    "clem_wings": [0,1,2,3,4,5,6,7],
    "fiona_wings": [0,1,2,3,4,5,6,7],
    "fiona_wings_2": [7,6,5,4,3,2,1,0],
    "ilaria_wings": [0,1,2,3,4,5,6,7],
    "ness_wings": [7,6,5,4,0,1,2,3],
    "ness_wings_2": [4,5,6,7,3,2,1,0],
    "ness_wings_3": [7,6,5,4,0,1,2,3],
    "ness_wings_4": [7,6,5,4,3,2,1,0],
    "sandra_wings": [0,1,2,3,4,5,6,7],
    "shaun_wings": [0,1,2,3,4,5,6,7],
    "tamblyn_wings": [6,7,4,5,3,2,1,0],
    "tamblyn_wings_2": [0,1,2,3,7,6,5,4],
    "tamblyn_wings_3": [0,1,2,3,7,6,5,4],
    "teresa_wings": [7,6,5,4,3,2,1,0],
    "teresa_wings_2": [0,1,2,3,4,5,6,7],
    "zoe_wings": [0,1,2,3,4,5,6,7]
}


class WingsInferenceDataLoader(BaseDataLoader):
    def __init__(self, folders_list, batch_size, resize_dims=(256, 256), shuffle=True, validation_split=0.0,
                 num_workers=1):
        self.dataset = WingDataInference(get_image_paths(folders_list), resize_dims=resize_dims)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class WingsTrainingDataLoader(BaseDataLoader):
    def __init__(self, tps_list, batch_size, resize_dims=(256, 256), shuffle=True, validation_split=0.0,
                 num_workers=1):
        image_paths, feature_coords = get_paths_from_tps_file(tps_list)
        self.dataset = WingDataTraining(image_paths, feature_coords, resize_dims=resize_dims, augment=True)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class WingDataInference(Dataset):

    def __init__(self, list_paths, resize_dims=(256, 256)):
        """
        Dataset for inference (does not require TPS)
        :param list_paths: List of image paths
        :param resize_dims: What size to make images
        """
        super().__init__()

        self.list_paths = list_paths
        self.resize_dims = resize_dims

        self.data_transform = transforms.Compose([
            transforms.Resize(resize_dims),
            transforms.ToTensor()])

        self.seq = iaa.Sequential([iaa.Resize(resize_dims)])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_paths)

    def __getitem__(self, index):
        """
        Get a data item
        :param index:
        :return: an image as tensor and it's path
        """
        # Select sample
        sample_path = self.list_paths[index]

        if not os.path.isfile(sample_path):
            print("{} is not a file/does not exist!".format(sample_path))
        # Load data and get label
        image = cv.imread(sample_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.normalize(image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC3)

        image_aug = self.seq(image=image)

        input_tensor = ((torch.tensor(image_aug)).permute(2, 0, 1))

        return input_tensor, sample_path


def get_image_paths(folders_list):
    """
    Get all images in a given folder
    :param data_path: The list of folders from which to scrape images
    :return: List of images in data_path
    """
    image_paths = []
    for data_path in folders_list:
        for ext in ('*.tif', '*.bmp', '*.png', '*.jpg'):
            for filename in Path(data_path).glob(ext):
                image_paths.append(str(filename))
    image_paths.sort()
    return image_paths


def get_inference_data(folders_list, resize_dims=(256, 256), device="cpu"):
    """
    Get inference dataset
    :param folders_list: List of folders from which to grab images
    :param resize_dims: Size to make images
    :param device: Device to process images on
    :return: Dataset for inference containing all images in folder from folders_list
    """
    data = WingDataInference(get_image_paths(folders_list), resize_dims=resize_dims, device=device)
    train_size = int(len(data))
    data_infer, data_test = random_split(data, [train_size, len(data) - train_size])
    return data_infer


class WingDataTraining(Dataset):

    def __init__(self, list_paths, labels, resize_dims=(256, 256), augment=False):
        """
        Dataset for training
        :param list_paths: List of image paths
        :param labels: List of keypoints
        :param resize_dims: What size to make images
        :param augment: Whether to augment images or not
        :param device: Which device to load tensors to
        """
        super().__init__()

        self.list_paths = list_paths
        self.labels = labels
        self.resize_dims = resize_dims
        self.keypoint_divisor = np.array([resize_dims[0], resize_dims[1], resize_dims[0], resize_dims[1],
                                          resize_dims[0], resize_dims[1], resize_dims[0], resize_dims[1],
                                          resize_dims[0], resize_dims[1], resize_dims[0], resize_dims[1],
                                          resize_dims[0], resize_dims[1], resize_dims[0], resize_dims[1]])
        self.augment = augment

        self.data_transform = transforms.Compose([
            transforms.Resize(resize_dims),
            transforms.ToTensor()])

        self.seq_basic = iaa.Sequential([iaa.Resize(resize_dims)])

        self.seq1 = iaa.Sequential([
            iaa.Affine(scale=(0.7, 1.0), mode='edge'),  # 'reflect'
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Resize(resize_dims)])

        self.seq2 = iaa.Sequential([
            iaa.Affine(rotate=(-60, 60), scale=(0.7, 1.1), mode='edge'),  # 'reflect'
            iaa.Crop(px=(0, 25)),  # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Flipud(0.5),
            iaa.Resize(resize_dims)])

    @staticmethod
    def add_noise(image, mean, var):
        row, col, ch = image.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    @staticmethod
    def np_to_keypoints(np_kpoints, image_size):
        np_kpoints = np_kpoints
        kps = [
            Keypoint(x=np_kpoints[0], y=image_size[0] - np_kpoints[1]),
            Keypoint(x=np_kpoints[2], y=image_size[0] - np_kpoints[3]),
            Keypoint(x=np_kpoints[4], y=image_size[0] - np_kpoints[5]),
            Keypoint(x=np_kpoints[6], y=image_size[0] - np_kpoints[7]),
            Keypoint(x=np_kpoints[8], y=image_size[0] - np_kpoints[9]),
            Keypoint(x=np_kpoints[10], y=image_size[0] - np_kpoints[11]),
            Keypoint(x=np_kpoints[12], y=image_size[0] - np_kpoints[13]),
            Keypoint(x=np_kpoints[14], y=image_size[0] - np_kpoints[15]),
        ]
        return kps

    def point_out_of_range(self, kpts):
        kpts_np = kpts.to_xy_array()
        in_range_x = (kpts_np[:, 0] >= 0).all() and (kpts_np[:, 0] < self.resize_dims[0]).all()
        in_range_y = (kpts_np[:, 1] >= 0).all() and (kpts_np[:, 1] < self.resize_dims[1]).all()
        out_of_range = not in_range_x or not in_range_y
        return out_of_range

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_paths)

    def __getitem__(self, index):
        """
        Generates one sample of data
        :param index: The index of the image in the list of paths
        :return: Returns image as tensor, keypoints as tensor and image path as string
        """
        # Select sample
        sample_path = self.list_paths[index]

        if not os.path.isfile(sample_path):
            print("{} is not a file/does not exist!".format(sample_path))
        # Load data and get label
        image = cv.imread(sample_path)
        if image is None:
            print("{} is not a valid image".format(sample_path))
        image_size = image.shape

        kps = self.np_to_keypoints(self.labels[index].flatten(), image_size)
        kpsoi = KeypointsOnImage(kps, shape=image.shape)
        #         image_aug, kpsoi_aug = self.seq(image=image, keypoints=kpsoi)
        if not self.augment:
            image_aug, kpsoi_aug = self.seq_basic(image=image, keypoints=kpsoi)
            image_aug = cv.cvtColor(image_aug, cv.COLOR_BGR2RGB)

        if self.augment:
            image_aug, kpsoi_aug = self.seq2(image=image, keypoints=kpsoi)
            out_of_range = self.point_out_of_range(kpsoi_aug, image_size)
            if out_of_range:
                image_aug, kpsoi_aug = self.seq1(image=image, keypoints=kpsoi)
            #             H: 0-179, S: 0-255, V: 0-255
            image_aug = cv.cvtColor(image_aug, cv.COLOR_BGR2HSV)
            add_hue = np.random.normal(0, 8)
            add_sat = np.random.normal(0, 10)
            add_val = np.random.normal(0, 0)
            #             print("h={}, s={}, v={}".format(add_hue, add_sat, add_val))
            image_aug[:, :, 0] = np.mod((image_aug[:, :, 0] + int(add_hue)), 180)
            image_aug[:, :, 1] = np.clip((image_aug[:, :, 1] + int(add_sat)), 0, 254)
            image_aug[:, :, 2] = np.clip((image_aug[:, :, 2] + int(add_val)), 0, 254)

            image_aug = cv.cvtColor(image_aug, cv.COLOR_HSV2RGB)

            variance = random.uniform(0, 80)
            image_aug = self.add_noise(image_aug, 0, variance)

        image_aug = cv.normalize(image_aug, None, alpha=0, beta=1,
                                 norm_type=cv.NORM_MINMAX, dtype=cv.CV_32FC3)

        input_tensor = ((torch.tensor(image_aug)).permute(2, 0, 1))  # self.normalize
        output_tensor = torch.tensor(kpsoi_aug.to_xy_array().flatten() / self.keypoint_divisor)

        return input_tensor, output_tensor, sample_path


def get_paths_from_tps_file(path_to_file, base_path, point_orders=default_point_orders):
    """
    Method to get array of image paths and array of keypoints from TPS files
    :param path_to_file: Path to a file that contains a list of TPS files
    :param base_path: The base path of the folder containing all wing photos, ie: "/storage/data_storage/wings/"
    :param point_orders: In which order the keypoints in each folder are
    :return Returns list of image paths and array of keypoints
    """
    data_files = pd.read_csv(path_to_file, header=None, delimiter="\n").values.flatten().tolist()

    image_paths = []
    feature_coords = []
    success_cnt = 0
    fail_cnt = 0

    for file in data_files:
        file_path = base_path + file
        #         print(file_path)
        folder_names = re.split('/|\n', file_path)
        point_order = point_orders[folder_names[5]]
        f = open(file_path, 'r')
        cnt = 0

        folder_path = os.path.dirname(file)
        img_feature_coords = []

        warning_given = False
        for line in f:
            str_in = re.split('=|\n', line)
            if str_in[0] == "SCALE" or str_in[0] == "LM" or str_in[0] == "ID":
                continue
            elif str_in[0] == "IMAGE":
                image_name = re.split('=|\n', line)
                if image_name[1][0] == ".":
                    image_name[1] = image_name[1][1:]
                #                     print("First character is dot: {}".format(image_name[1]))
                image_path = (base_path + folder_path + "/" + image_name[1]).strip()
                if os.path.isfile(image_path) and len(img_feature_coords) == 8:
                    image_paths.append(image_path)

                    features = np.asarray(img_feature_coords, dtype=np.float32, order='C')
                    permuted_features = []
                    for i in range(0, len(features), 1):
                        permuted_features.append(img_feature_coords[point_order[i]])
                    permuted_features = np.asarray(permuted_features, dtype=np.float32, order='C')
                    feature_coords.append(permuted_features)
                    success_cnt += 1
                else:
                    if not warning_given:
                        #                         print("==================================")
                        print("Issue with {} (has {} coordinates)".format(
                            image_path, len(img_feature_coords)))
                        #                         print(img_feature_coords)
                        #                         print("{}, {}".format(image_name, line))
                        warning_given = True

                    fail_cnt += 1
                #                     print(img_feature_coords)
                img_feature_coords = []
            else:
                coords_str = str.split(line)
                img_feature_coords.append(coords_str)
    print("Success/fail = {}/{}".format(success_cnt, fail_cnt))
    return image_paths, feature_coords


def get_training_data(data_list_path, resize_dims=(256,256), device="cpu", train_ratio=1.0):
    """
    Get training and validation data
    :param data_list_path: The path to the file that contains list of TPS files
    :param resize_dims: The size that the images should be
    :param device: The device to train on
    :param train_ratio: Ration of training/validation
    :return: training and validation data
    """
    image_paths, feature_coords = get_paths_from_tps_file(data_list_path)
    data = WingDataTraining(image_paths, feature_coords, resize_dims=resize_dims, augment=True, device=device)
    train_size = int(len(data) * train_ratio)
    data_train, data_test = random_split(data, [train_size, len(data) - train_size])

    return data_train, data_test
