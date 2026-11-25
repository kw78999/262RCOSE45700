"""Split independent Conv output channels while retaining every input channel."""
import copy

import torch


class OutputPartitionConv(torch.nn.Module):
    def __init__(self, original, chunk=128):
        super().__init__()
        if type(chunk) is not int or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        if original.groups != 1 or original.padding_mode != "zeros":
            raise ValueError("Output partitioning requires a dense, zero-padded Conv2d")
        self.parts = torch.nn.ModuleList()
        for start in range(0, original.out_channels, chunk):
            end = min(original.out_channels, start + chunk)
            part = torch.nn.Conv2d(original.in_channels, end - start, original.kernel_size,
                                   stride=original.stride, padding=original.padding,
                                   dilation=original.dilation, bias=original.bias is not None,
                                   dtype=original.weight.dtype, device=original.weight.device)
            with torch.no_grad():
                part.weight.copy_(original.weight[start:end])
                if original.bias is not None:
                    part.bias.copy_(original.bias[start:end])
            self.parts.append(part)
        self.train(original.training)

    def forward(self, inputs):
        return torch.cat([part(inputs) for part in self.parts], dim=1)


def partition_stage4(model, chunk=128, *, inplace=False):
    model = model if inplace else copy.deepcopy(model)
    replaced = []
    for index in range(3):
        path = f"layer4.{index}.conv2"
        conv = model.get_submodule(path)
        if not isinstance(conv, torch.nn.Conv2d) or tuple(conv.weight.shape) != (512, 512, 3, 3):
            raise ValueError(f"Unexpected ResNet50 module at {path}")
        split = OutputPartitionConv(conv, chunk)
        model.layer4[index].conv2 = split
        replaced.append({"module": path, "weight_shape": list(conv.weight.shape),
                         "stride": list(conv.stride), "padding": list(conv.padding),
                         "part_weight_shapes": [list(part.weight.shape) for part in split.parts]})
    return model, {"axis": "output_channels", "chunk_size": chunk,
                   "all_input_channels_in_every_part": True, "combine": "concat_dim1",
                   "modules": replaced}


class _BatchSliced:
    """Forward of one module run on consecutive batch slices of at most `chunk`, then concatenated."""

    def __init__(self, forward, chunk):
        self.inner, self.batch_chunk = forward, chunk

    def __call__(self, inputs):
        if inputs.shape[0] <= self.batch_chunk:
            return self.inner(inputs)
        return torch.cat([self.inner(part) for part in inputs.split(self.batch_chunk, dim=0)], dim=0)


def partition_batch(model, module_paths, chunk, *, inplace=False):
    """Run each listed module (Conv, pool, activation or a whole residual block) on batch slices.

    Samples are independent in eval mode, so the output equals the original module. Only the
    instance `forward` is replaced: parameter names and values, and the state fingerprint, stay.
    """
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    model = model if inplace else copy.deepcopy(model)
    paths = sorted(set(module_paths))
    nested = [p for p in paths if any(p.startswith(q + ".") for q in paths if q != p)]
    if nested:
        raise ValueError(f"Module paths are nested inside other listed modules: {nested}")
    replaced = []
    for path in paths:
        module = model.get_submodule(path)
        if module.training or any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                  for m in module.modules()):
            raise ValueError(f"{path}: batch slicing is exact only for eval modules without BatchNorm")
        if isinstance(module.forward, _BatchSliced):
            raise ValueError(f"{path} is already batch-partitioned")
        module.forward = _BatchSliced(module.forward, chunk)
        replaced.append({"module": path, "type": type(module).__name__})
    return model, {"axis": "batch", "chunk_size": chunk, "combine": "concat_dim0", "modules": replaced}
