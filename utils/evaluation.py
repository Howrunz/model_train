import numpy as np
from medpy.filter.binary import largest_connected_component

def evaluate_all(predict, truth):
    dice = []
    other = []
    for i in range(len(predict)):
        pred = np.array(predict[i])
        pred = np.round(pred).astype(int)
        pred = largest_connected_component(pred)
        true = np.array(truth[i])
        true = np.round(true).astype(int)
        dice.append(np.sum(pred[true == 1]) * 2.0 + 1 / (np.sum(pred) + np.sum(true) + 1))
        other.append(evaluate(pred, true))
    dice_result = np.mean(dice)
    other_result = np.mean(other, axis=0)
    return dice_result, other_result

def evaluate(predict, truth):
    smooth = 1
    pred_pos = predict
    pred_neg = 1 - pred_pos
    true_pos = truth
    true_neg = 1 - true_pos

    TP = np.sum(true_pos * pred_pos)
    TN = np.sum(true_neg * pred_neg)
    FP = np.sum(true_neg * pred_pos)
    FN = np.sum(true_pos * pred_neg)

    recall = (TP + smooth) / (TP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    accuracy = (TP + TN + smooth) / (TP + TN + FP + FN + smooth)
    sensitivity = (TP + smooth) / (TP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)
    F1_score = (2 * recall * precision) / (recall + precision)

    return [recall, precision, accuracy, sensitivity, specificity, F1_score]
