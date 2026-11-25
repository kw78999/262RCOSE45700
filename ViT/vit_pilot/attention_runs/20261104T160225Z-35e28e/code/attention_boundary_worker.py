"""Reuse the worker with a user-code boundary around each unchanged MHA.

The adapter replacement exists only in this isolated process, during preparation.
It changes no SDK/Torch functions and installs no diagnostic hooks in performance.
"""
import torch
import vit
from attention_worker import main


class AttentionBoundary(torch.nn.Module):
    def __init__(self, original_attention):
        super().__init__()
        self.original_attention = original_attention

    def forward(self, query, key, value, **kwargs):
        return self.original_attention(query, key, value, **kwargs)


original_prepare = vit.prepare_model_inputs


def prepare_with_attention_boundaries(*args, **kwargs):
    model, inputs = original_prepare(*args, **kwargs)
    parameters_before = {id(p) for p in model.parameters()}
    for block in model.encoder.layers:
        block.self_attention = AttentionBoundary(block.self_attention).eval()
    if {id(p) for p in model.parameters()} != parameters_before:
        raise RuntimeError('Boundary wrapping must retain the exact original Parameter objects')
    # No forward. The existing worker now compiles these user-code boundaries.
    return model, inputs


if __name__ == '__main__':
    vit.prepare_model_inputs = prepare_with_attention_boundaries
    raise SystemExit(main())
