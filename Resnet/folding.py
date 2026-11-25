"""Explicit eval-only Conv/BatchNorm folding, performed in FP32 before BF16."""
import copy
import hashlib

import torch
from torch.nn.utils import fuse_conv_bn_eval


def fold_batchnorm(model):
    """Return a copied model and a complete list of folded Conv2d/BN2d pairs.

    Supports ResNet's named convN/bnN pairs and Sequential downsample pairs.
    Refuses unmatched BN layers rather than silently leaving CPU-only operators.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Conv/BN folding requires every module to be in eval mode")
    if any(tensor.dtype != torch.float32 for tensor in model.parameters()):
        raise ValueError("Fold in FP32, then convert the resulting model to BF16")
    folded = copy.deepcopy(model)
    pairs = []
    for parent_path, parent in list(folded.named_modules()):
        candidates = []
        for index in (1, 2, 3):
            conv_name, bn_name = f"conv{index}", f"bn{index}"
            if isinstance(getattr(parent, conv_name, None), torch.nn.Conv2d) and isinstance(
                    getattr(parent, bn_name, None), torch.nn.BatchNorm2d):
                candidates.append((conv_name, bn_name))
        if isinstance(parent, torch.nn.Sequential):
            children = list(parent.named_children())
            candidates.extend((left_name, right_name)
                              for (left_name, left), (right_name, right) in zip(children, children[1:])
                              if isinstance(left, torch.nn.Conv2d) and isinstance(right, torch.nn.BatchNorm2d))
        for conv_name, bn_name in candidates:
            conv, bn = getattr(parent, conv_name), getattr(parent, bn_name)
            if bn.running_mean is None or bn.running_var is None:
                raise ValueError(f"Missing BatchNorm running statistics: {parent_path}.{bn_name}")
            setattr(parent, conv_name, fuse_conv_bn_eval(conv, bn))
            setattr(parent, bn_name, torch.nn.Identity().eval())
            prefix = f"{parent_path}." if parent_path else ""
            pairs.append({"conv": prefix + conv_name, "batchnorm": prefix + bn_name})
    remaining = [name for name, module in folded.named_modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if remaining:
        raise ValueError(f"Unmatched BatchNorm layers: {remaining}")
    return folded, {"name": "conv_bn_eval_folded", "api": "torch.nn.utils.fuse_conv_bn_eval",
                    "fold_dtype": "float32", "cast_after_fold": "bfloat16",
                    "pair_count": len(pairs), "pairs": pairs, "remaining_batchnorm": remaining}


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
