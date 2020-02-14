import yaml
import torch
import torchvision

def print_model_summary(model):
    for name, parameter in model.named_paraneters():
        print(name, parameter.numel(), parameter.requires_grad)
    return None

def save_weight(model, model_path, train_loss, valid_loss, epoch, step):
    torch.save(
        {
            'model': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, str(model_path)
    )

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        result = f.read()
        config = yaml.load(result, Loader=yaml.FullLoader)
    return config