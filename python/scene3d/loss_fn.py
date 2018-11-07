from scene3d import torch_utils
import torch
import torch.nn.functional as F
import numpy as np


def undo_log_depth(d):
    return (2.0 ** d) - 0.5


def loss_calc(pred, target):
    assert pred.shape[1] > 1
    mask = ~torch.isnan(target)
    return (torch.log2(target[mask] + 0.5) - pred[mask]).abs().mean()


def loss_calc_overhead_single_log(pred, target):
    assert pred.shape[1] == 1
    mask = ~torch.isnan(target)
    return (torch.log2(target[mask] + 0.5) - pred[mask]).abs().mean()


def loss_calc_overhead_single_raw(pred, target):
    assert pred.shape[1] == 1
    mask = ~torch.isnan(target)
    return (target[mask] - pred[mask]).abs().mean()


def loss_calc_single_depth(pred, target):
    assert pred.shape[1] == 1
    target_single_depth = target[:, 0, None] - target[:, 1, None]
    mask = ~torch.isnan(target_single_depth)
    return (torch.log2(target_single_depth[mask] + 0.5) - pred[mask]).abs().mean()


def loss_calc_classification(pred, target, ignore_index=None):
    target = target.long().cuda()
    ignore_index_ = -100
    if ignore_index:
        ignore_index_ = ignore_index
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_).cuda()
    return criterion(pred, target)


def compute_masked_smooth_l1_loss(pred, target, apply_log_to_target=True):
    assert pred.dim() == 4
    assert target.size() == pred.size(), (target.shape, pred.shape)

    valid = torch.isfinite(target)  # (B, C, H, W)

    if apply_log_to_target:
        t = torch.log2(target + 0.5)
    else:
        t = target

    l1 = (t - pred).masked_fill(~valid, 0).abs()  # (B, C, H, W) contains nan values.
    l2 = torch.pow(l1, 2.0)  # (B, C, H, W) contains nan values
    select = l1 >= 1.0

    # per-image mean huber loss
    loss = (0.5 * l2).masked_scatter(select, l1[select] - 0.5).view(l2.shape[0], l2.shape[1], -1).sum(dim=2)  # (B, C) float32
    assert loss.dtype == torch.float32

    area = valid.view(valid.shape[0], valid.shape[1], -1).sum(dim=2)  # (B, C) int64
    assert area.dtype == torch.int64
    assert area.shape == loss.shape

    loss = (loss / (area.type(loss.dtype) + 1e-3)).mean()

    return loss


def compute_masked_surface_normal_loss(pred, target, use_inverse_cosine=False):
    assert pred.dim() == 4
    assert pred.shape[1] == 3
    assert target.size() == pred.size(), (target.shape, pred.shape)

    finite_mask = torch.isfinite(target)  # (B, 1, H, W)

    pred_normalized = F.normalize(pred, p=2, dim=1)

    dotproduct = (pred_normalized * target.masked_fill(~finite_mask, 0)).sum(dim=1, keepdim=True)  # contains nans

    valid_values = dotproduct[finite_mask[:, :1]]  # n'n

    if use_inverse_cosine:
        lmda = 4
        loss = (-valid_values + lmda * torch.acos(torch.clamp(valid_values, -0.99, 0.99))).mean()
    else:
        loss = -valid_values.mean()

    return loss  # every pixel contributes equally


def eval_mode_compute_masked_surface_normal_error(pred, target, use_angular_error=True, return_error_map=False, return_pred_normalized=False):
    assert pred.dim() == 4
    assert pred.shape[1] == 3
    assert target.size() == pred.size(), (target.shape, pred.shape)

    finite_mask = torch.isfinite(target)  # (B, 1, H, W)

    pred_normalized = F.normalize(pred, p=2, dim=1)

    dotproduct = (pred_normalized * target.masked_fill(~finite_mask, 0)).sum(dim=1, keepdim=True)  # contains nans

    valid_values = dotproduct[finite_mask[:, :1]]  # n'n

    if use_angular_error:
        error = torch.acos(torch.clamp(valid_values, -0.999999, 0.999999)).mean()
    else:
        error = -valid_values.mean()

    error = torch_utils.recursive_torch_to_numpy(error)

    ret = {
        'error': error.item(),
        'pred_normalized': torch_utils.recursive_torch_to_numpy(pred_normalized),
    }

    if return_error_map:
        error_map = torch_utils.recursive_torch_to_numpy(dotproduct)
        finitemask_np = torch_utils.recursive_torch_to_numpy(finite_mask)

        error_map[~finitemask_np[:, :1].astype(np.bool)] = np.nan
        with np.errstate(invalid='ignore'):
            error_map[finitemask_np[:, :1].astype(np.bool)] = np.arccos(error_map[finitemask_np[:, :1].astype(np.bool)])

        ret['error_map'] = error_map

    # Keys: error, error_map
    return ret
