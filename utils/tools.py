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