"""CPU structural/numerical check of the exact folding used by both NPU modes."""
import argparse
import copy
import sys
import traceback
from pathlib import Path

import torch
from torchvision.models import resnet50

from Resnet.analysis import compare_tensors
from Resnet.artifacts import write_json
from Resnet.folding import fold_batchnorm, state_fingerprint
from Resnet.models import WEIGHTS, load_model, make_input, weight_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"status": "running", "scope": "CPU transformation validation, not NPU performance"}
    try:
        original = resnet50(weights=WEIGHTS, progress=False).eval()
        folded, metadata = fold_batchnorm(original)
        assert metadata["pair_count"] == 53
        assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in original.modules()) == 53
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())
        inputs = make_input(1)
        with torch.inference_mode():
            reference = original(inputs.float())
            transformed = folded(inputs.float())
            torch.testing.assert_close(reference, transformed, rtol=1e-4, atol=1e-4)
            original_bf16 = original.bfloat16()(inputs)
            folded_bf16 = folded.bfloat16()(inputs)
        prepared = load_model(bn_folded=True)
        assert state_fingerprint(folded) == prepared._benchmark_transform["state_sha256"]
        result.update(status="ok", weights=weight_metadata(), transform=prepared._benchmark_transform,
                      fp32_assert_close={"rtol": 1e-4, "atol": 1e-4, "passed": True},
                      fp32_original_vs_folded=compare_tensors(reference, transformed),
                      bf16_original_vs_folded=compare_tensors(original_bf16, folded_bf16),
                      fp32_original_vs_folded_bf16=compare_tensors(reference, folded_bf16))
        code = 0
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        code = 1
    write_json(args.output, result)
    print(result["status"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
