import torch
import time
from MinkowskiEngine.utils import sparse_quantize


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_executed = time.time() - start_time
        if hasattr(args[0], 'scalar_summaries'):  # args[0] is self
            args[0].scalar_summaries['resources/time/{}'.format(func.__name__)] += [time_executed]
        return result

    return wrapper


def quantize(coord: torch.Tensor, voxel_size) -> torch.Tensor:
    # quantize to nearest neighbor
    round_coord = torch.round(coord / voxel_size).cpu()
    return sparse_quantize(
        round_coord, return_index=False, quantization_size=1
    )


def downsample(x: torch.tensor, sample_num: int, deterministic=False) -> torch.tensor:
    '''
    Args:
        x: torch tensor of N x d
        sample_num: number of samples to downsample to
        deterministic: whether the procedure is deterministic
    Returns:
        downsampled output of torch tensor {sample_num} x d
    '''
    if x.shape[0] == 0:
        return torch.zeros(sample_num, 3)

    if x.shape[0] < sample_num:
        multiplier = (int(sample_num) // x.shape[0])
        x_multiply = torch.cat((x, ) * multiplier, dim=0)
        sample_num -= multiplier * x.shape[0]
        return torch.cat([downsample(x, sample_num, deterministic), x_multiply], dim=0)

    rand_idx = torch.arange(x.shape[0]) \
        if deterministic else torch.randperm(x.shape[0])
    keep_idx = rand_idx[:sample_num]
    return x[keep_idx, :]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
