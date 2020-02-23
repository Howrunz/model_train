import time
from utils.evaluation import evaluate_all

import torch
import torch.nn as nn
import torch.nn.functional as F

def validation(model:nn.Module, criterion, valid_loader, device, writer, epoch):
    val_loss = 0.
    validation_pred = []
    validation_true = []

    model.eval()
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
        print('validation epoch time: ', end_time)
    val_dice, other = evaluate_all(validation_pred, validation_true)
    recall, precision, accuracy, sensitivity, specificity, F1_score = other

    writer.add_scalar('evaluation/recall', recall, epoch)
    writer.add_scalar('evaluation/precision', precision, epoch)
    writer.add_scalar('evaluation/accuracy', accuracy, epoch)
    writer.add_scalar('evaluation/sensitivity', sensitivity, epoch)
    writer.add_scalar('evaluation/specificity', specificity, epoch)
    writer.add_scalar('evaluation/F1_score', F1_score, epoch)
    print('validation dice: %4f' % val_dice)
    print('validation F1_score: %4f' % F1_score)
    print('validation recall: %4f' % recall)
    print('validation precision: %4f' % precision)
    print('validation accuracy: %4f' % accuracy)
    print('validation sensitivity: %4f' % sensitivity)
    print('validation specificity: %4f' % specificity)

    return val_loss