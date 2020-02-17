import numpy as np
from medpy.filter.binary import largest_connected_component

def evaluate_dice(predict, truth):
    dice = []
    for i in range(len(predict)):
        pred = np.array(predict[i])
        pred = np.round(pred).astype(int)
        pred = largest_connected_component(pred)
        true = np.array(truth[i])
        true = np.round(true).astype(int)
        dice.append(np.sum(pred[true == 1]) * 2.0 + 1 / (np.sum(pred) + np.sum(true) + 1))
    dice_result = np.mean(dice)
    return dice_result