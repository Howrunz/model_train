import torch
import torchvision

def print_model_summary(model):
    for name, parameter in model.named_paraneters():
        print(name, parameter.numel(), parameter.requires_grad)
    return None