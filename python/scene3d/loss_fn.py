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
