"""Only the requested torchvision ViT adapter and deterministic assets."""
from pathlib import Path
import torch
from common import sha256, tensor_hash, write_json

MODEL_ID = 'torchvision.models.vit_b_16'
WEIGHTS_ID = 'ViT_B_16_Weights.IMAGENET1K_V1'


def prepare_assets(run, image_paths=None):
    from torchvision.models import ViT_B_16_Weights
    w = ViT_B_16_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir())/'checkpoints'/Path(w.url).name
    # Official URL / default torch hub directory; check_hash verifies URL hash prefix.
    if not checkpoint.exists():
        state = w.get_state_dict(progress=True, check_hash=True)
        del state
    weight_hash = sha256(checkpoint)
    if not weight_hash.startswith(Path(w.url).stem.split('-')[-1]):
        raise RuntimeError('Official checkpoint SHA256 prefix mismatch')
    if image_paths:
        from PIL import Image
        paths = [Path(p).resolve() for p in image_paths]
        if len(paths) != 4 or len(set(paths)) != 4:
            raise ValueError('Provide exactly four distinct fixed images')
        samples = []
        for p in paths:
            with Image.open(p) as im:
                samples.append(w.transforms()(im.convert('RGB')))
        x = torch.stack(samples)
        ids = [str(p) for p in paths]
        source_hashes = [sha256(p) for p in paths]
    else:
        generator = torch.Generator(device='cpu').manual_seed(42)
        x = torch.randn(4, 3, 224, 224, generator=generator, device='cpu', dtype=torch.float32)
        ids = [f'synthetic_seed42_{i}' for i in range(4)]
        source_hashes = None
    torch.save(x, run/'inputs.pt')
    xb = x.to(torch.bfloat16)
    info = dict(model_id=MODEL_ID, weights_id=WEIGHTS_ID, weights_url=w.url,
                weights_file=str(checkpoint), weights_sha256=weight_hash,
                synthetic_input=not bool(image_paths), real_image_validation=bool(image_paths), seed=42,
                unique_samples=4, input_ids=ids, source_hashes=source_hashes,
                input_file_sha256=sha256(run/'inputs.pt'), stored_dtype=str(x.dtype), dtype='torch.bfloat16',
                sample_bf16_sha256=[tensor_hash(t) for t in xb],
                batch_hashes={str(b):tensor_hash(xb[:b]) for b in [1,4]},
                input_shapes={str(b):[b,3,224,224] for b in [1,4]},
                tokens_per_image=197, preprocessing=str(w.transforms()) if image_paths else 'none; fixed random normal tensor')
    write_json(run/'assets.json', info)
    return info


def prepare_model_inputs(run, batches, device):
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from common import read_json
    assets = read_json(run/'assets.json')
    if sha256(assets['weights_file']) != assets['weights_sha256']:
        raise RuntimeError('Checkpoint changed since asset preparation')
    if sha256(run/'inputs.pt') != assets['input_file_sha256']:
        raise RuntimeError('Stored inputs changed')
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(dtype=torch.bfloat16)
    cpu = torch.load(run/'inputs.pt', map_location='cpu', weights_only=True).to(torch.bfloat16)
    for b in batches:
        if tensor_hash(cpu[:b]) != assets['batch_hashes'][str(b)]:
            raise RuntimeError('Input tensor hash mismatch')
    # No forward here. Input creation and all device transfers precede timing.
    model = model.to(device)
    inputs = {b:cpu[:b].contiguous().to(device) for b in batches}
    return model, inputs
