import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np


def load_model(path_to_model):
    print('Loading model...')
    w_net_model = module_arch.wingnet()
    w_net_model.load_state_dict(torch.load(path_to_model))

    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device


def pass_forward(data, model, device, norm_factor):
    output_valid = model(data.to(device, dtype=torch.float)).cpu().detach().numpy()
    output_valid = np.squeeze(output_valid) * norm_factor
    return output_valid


RESIZE = (256, 256)
KPT_DIV = np.array([RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1],
                    RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1], RESIZE[0], RESIZE[1]])

model = load_model("/home/timo/Data2/wingNet_models/wings_resnet34_weights")

device = get_device(True)
model = model.to(device)
model.eval()

folders_list = ["/home/timo/Data2/wings/clem_wings/clem_wings/A.F1_SN.xch"]

dl = module_data.WingsInferenceDataLoader(folders_list, 32, resize_dims=(256, 256), shuffle=False, validation_split=0.0,
                                          num_workers=1)

for batch_idx, (data, name) in enumerate(dl):

    keypoints = pass_forward(data, model, device, KPT_DIV)
    print(keypoints.shape)
    print(data.shape)
    print(data.type())
    img_in = data[0].numpy()
    img_in = np.transpose(img_in, (1, 2, 0))
    plt.imshow(img_in)
    plt.scatter(keypoints[0][::2], keypoints[0][1::2], c='r', marker='x')
    plt.show()
