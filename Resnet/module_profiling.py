"""Capture actual CPU activations and replay individual FX operations on NPU."""
import copy
import time

import torch
from torch.fx import Interpreter, symbolic_trace


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    return value


def tensor_shapes(value):
    return map_tensors(value, lambda tensor: list(tensor.shape))


class ReplayNode(torch.nn.Module):
    def __init__(self, kind, operation):
        super().__init__()
        self.kind = kind
        self.operation = copy.deepcopy(operation) if kind == "call_module" else operation

    def forward(self, *args, **kwargs):
        if self.kind == "call_method":
            return getattr(args[0], self.operation)(*args[1:], **kwargs)
        return self.operation(*args, **kwargs)


def capture_operations(model, input_tensor):
    """Include residual adds, flatten, concat and repeated calls to shared ReLU."""
    graph = symbolic_trace(model)
    operations = []

    class Capture(Interpreter):
        def run_node(self, node):
            if node.op not in ("call_module", "call_function", "call_method"):
                return super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            # Clone BEFORE execution: an in-place ReLU can mutate its own input.
            args, kwargs = map_tensors((args, kwargs), lambda tensor: tensor.detach().clone())
            output = super().run_node(node)
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected tensor output at {node.name}")
            target = graph.get_submodule(node.target) if node.op == "call_module" else node.target
            stack = list(node.meta.get("nn_module_stack", {}).keys())
            module = str(node.target) if node.op == "call_module" else (stack[-1] if stack else "<model>")
            operator = type(target).__name__ if node.op == "call_module" else getattr(target, "__name__", str(target))
            operations.append({"node": node.name, "module": module, "operator": operator,
                               "kind": node.op, "module_stack": stack,
                               "identity": isinstance(target, torch.nn.Identity),
                               "args": args, "kwargs": kwargs,
                               "reference": output.detach().clone(), "operation": target})
            return output

    with torch.inference_mode():
        output = Capture(graph).run(input_tensor)
    return operations, output


def inventory(operations):
    return [{"index": index, **{key: item[key] for key in
             ("node", "module", "operator", "kind", "module_stack", "identity")},
             "input_shapes": tensor_shapes(item["args"]),
             "output_shape": list(item["reference"].shape)} for index, item in enumerate(operations)]


def measure_operation(function, args, kwargs, device):
    from Resnet.runtime import synchronize
    # A new input for every call keeps in-place operations reproducible.
    device_args, device_kwargs = map_tensors((args, kwargs), lambda tensor: tensor.to(device, copy=True))
    synchronize(device)
    started = time.perf_counter_ns()
    output = function(*device_args, **device_kwargs)
    synchronize(device)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return elapsed, output
