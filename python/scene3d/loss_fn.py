import torch


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


def loss_calc_classification(pred, target):
    target = target.long().cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
    return criterion(pred, target)


def compute_masked_smooth_l1_loss(pred, target, apply_log_to_target=True):
    assert pred.dim() == 4
    assert target.size() == pred.size()

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
