import time

import torch
import torch.nn as nn
import torch.nn.functional as F

def validation(model:nn.Module, criterion, valid_loader, device):
    val_loss = 0.
    validation_pred = []
    validation_true = []
    with torch.no_grad():
        start_time = time.time()
        for image, mask in valid_loader:
            image = image.to(device)
            mask = mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            val_loss += loss.item()
            output_np = output.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()
            validation_pred.extend([output_np[s] for s in range(output_np.shape[0])])
            validation_true.extend([mask_np[s] for s in range(mask_np.shape[0])])
        end_time = time.time() - start_time
        print('vaildation epoch time: ', end_time)
        # TODO
        # calcuate criticize
    return val_loss #, criticize