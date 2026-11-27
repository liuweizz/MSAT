"""Exact pooled engine adapted from the validated RADIOlogy evaluation.
Function bodies retained. See third-party notices for provenance.
"""
import numpy as np
import torch
from scipy.ndimage import label as cc_label

def gt_region_weights(masks):
    weights = np.zeros(masks.shape, np.float64)
    region_count = 0
    for i, mask in enumerate(masks):
        labels, count = cc_label(mask, np.ones((3, 3), int))
        counts = np.bincount(labels.ravel(), minlength=count+1)
        inv = np.zeros(len(counts), np.float64)
        inv[1:] = 1. / counts[1:]
        weights[i] = inv[labels]
        region_count += count
    return weights, region_count


@torch.inference_mode()
def pooled_metrics(scores, masks, region_weights, nregions, device, trapezoid):
    values, order = torch.sort(torch.from_numpy(scores.ravel()).to(device), descending=True)
    labels = torch.from_numpy(masks.ravel().astype(np.float64)).to(device)[order]
    pro_weight = torch.from_numpy(region_weights.ravel()).to(device)[order]
    keep = torch.ones_like(values, dtype=torch.bool)
    keep[:-1] = values[1:] != values[:-1]
    thresholds = values[keep]
    del values, order
    tp = torch.cumsum(labels, 0)[keep]
    fp = torch.cumsum(1-labels, 0)[keep]
    pro = torch.clamp(torch.cumsum(pro_weight, 0)[keep] / nregions, max=1.)
    del labels, pro_weight, keep
    npos = float(masks.sum()); nneg = float(masks.size - masks.sum())
    recall, fpr = tp / npos, fp / nneg
    precision = tp / (tp + fp)
    zero, one = tp.new_zeros(1), tp.new_ones(1)
    auroc = trapezoid(torch.cat([zero, fpr, one]), torch.cat([zero, recall, one]))
    ap = (precision * torch.diff(torch.cat([zero, recall]))).sum()
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall), nan=0.).max()
    pro_fpr = torch.cat([zero, fpr, one]); pro_values = torch.cat([zero, pro, one])
    aupro = trapezoid(pro_fpr, pro_values, x_max=.3) / .3
    # right=True chooses the last/highest PRO at an exact repeated FPR.
    target = pro_fpr.new_tensor(.01)
    upper = int(torch.searchsorted(pro_fpr, target, right=True))
    lower = upper-1
    exact = bool(pro_fpr[lower] == target)
    if exact:
        upper = lower
    low_fpr, high_fpr = float(pro_fpr[lower]), float(pro_fpr[upper])
    low_pro, high_pro = float(pro_values[lower]), float(pro_values[upper])
    weight = 0. if exact else (.01-low_fpr)/(high_fpr-low_fpr)

    def threshold_at(index):
        if index == 0:
            return '+inf'
        if index > len(thresholds):
            return '-inf'
        return float(thresholds[index-1])

    point = {'target_fpr': .01, 'lower_threshold': threshold_at(lower),
             'upper_threshold': threshold_at(upper), 'lower_fpr': low_fpr,
             'upper_fpr': high_fpr, 'lower_pro': low_pro, 'upper_pro': high_pro,
             'interpolation_weight_upper': weight, 'exact_fpr_hit': exact,
             'threshold_rule': 'score >= threshold; lower/upper ordered by FPR',
             'not_a_deployment_threshold': True}
    result = {'pixel_auroc': float(auroc), 'pixel_ap': float(ap), 'pixel_f1': float(f1),
              'aupro': float(aupro), 'pro_at_fpr_0p01': low_pro + weight*(high_pro-low_pro),
              'pro1_operating_point': point, 'positive_pixels': int(npos),
              'negative_pixels': int(nneg), 'gt_regions_8_connected': nregions}
    del tp, fp, pro, thresholds, recall, fpr, precision, pro_fpr, pro_values
    return result

