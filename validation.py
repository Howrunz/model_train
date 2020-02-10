import time

import torch
import torch.nn as nn
import torch.nn.functional as F

def validation(model:nn.Module, criterion, valid_loader, device):
    with torch.no_grad():
        start_time = time.time()
        for image, mask in valid_loader:
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
        end_time = time.time() - start_time
        print('vaildation epoch time: ', end_time)
    return loss