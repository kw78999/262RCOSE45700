"""One explicit pretrained model and reproducible, synthetic inputs."""
import hashlib
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def checkpoint_path():
    return Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS.url.rsplit("/", 1)[1]


def load_model(seed=2026, bn_folded=False, output_split_channels=None):
    if output_split_channels is not None and (not bn_folded or output_split_channels != 128):
        raise ValueError("This experiment uses BN folding followed by output-channel split128")
    torch.manual_seed(seed)
    # The builder downloads the official checkpoint if missing and verifies its hash.
    model = resnet50(weights=WEIGHTS, progress=False).eval()
    transform = {"name": "identity", "pair_count": 0}
    if bn_folded:
        from Resnet.folding import fold_batchnorm
        model, transform = fold_batchnorm(model)
        if transform["pair_count"] != 53:
            raise RuntimeError(f"Expected 53 ResNet50 Conv/BN pairs, got {transform['pair_count']}")
    if output_split_channels is not None:
        from Resnet.partitioning import partition_stage4
        model, partition = partition_stage4(model, output_split_channels, inplace=True)
        transform = {**transform, "name": "bn_folded_output_split128", "output_partition": partition}
    model = model.to(dtype=torch.bfloat16)
    if bn_folded:
        from Resnet.folding import state_fingerprint
        transform["state_sha256"] = state_fingerprint(model)
    model._benchmark_transform = transform
    return model


def weight_metadata():
    path = checkpoint_path()
    return {
        "model": "torchvision.models.resnet50",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "url": WEIGHTS.url,
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
    }


def make_input(batch, image_size=224, seed=2026):
    generator = torch.Generator(device="cpu").manual_seed(seed + batch * 1009 + image_size)
    # Synthetic normal data, NOT an ImageNet accuracy evaluation.
    return torch.randn(batch, 3, image_size, image_size, generator=generator).to(torch.bfloat16)
