import torch


def move_tensors_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {k: move_tensors_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_cpu(item) for item in data}
    else:
        return data
