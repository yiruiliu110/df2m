
import torch


def standardize(inputs):
    means = torch.mean(inputs, dim=[1], keepdim=True).float()
    std = torch.mean(torch.std(inputs, dim=[1, 2], keepdim=True), dim=0, keepdim=True).float()
    return ((inputs-means) / std).float(), means, std
