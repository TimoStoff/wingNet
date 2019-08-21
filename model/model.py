#Learning
import torch
import torchvision.models as models


def get_wingnet_model():

    output_linear = torch.nn.Linear(512, 16, bias=True)
    model = models.resnet34(pretrained=True)
    model.fc = output_linear
    return model
