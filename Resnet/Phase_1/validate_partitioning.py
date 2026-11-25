"""Validate the full BN-folded + output-split128 model on CPU before NPU runs."""
import argparse
import json
import traceback
from pathlib import Path

import torch
from torchvision.models import resnet50

from Resnet.analysis import compare_tensors
from Resnet.artifacts import write_json
from Resnet.folding import fold_batchnorm, state_fingerprint
from Resnet.models import WEIGHTS, load_model, make_input, weight_metadata
from Resnet.partitioning import OutputPartitionConv, partition_stage4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = {"status": "running", "scope": "CPU transformation validation; not NPU inference performance"}
    try:
        original = resnet50(weights=WEIGHTS, progress=False).eval()
        folded, folding = fold_batchnorm(original)
        partitioned, partition = partition_stage4(folded)
        assert sum(isinstance(m, OutputPartitionConv) for m in partitioned.modules()) == 3
        assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in partitioned.modules())
        comparisons = []
        for batch in (1, 4):
            inputs = make_input(batch)
            with torch.inference_mode():
                a, b = folded(inputs.float()), partitioned(inputs.float())
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
            comparisons.append({"batch": batch, "fp32_folded_vs_split": compare_tensors(a, b)})
        original = original.bfloat16()
        folded = folded.bfloat16()
        partitioned = partitioned.bfloat16()
        for row in comparisons:
            batch = row["batch"]
            inputs = make_input(batch)
            with torch.inference_mode():
                baseline, output, unchanged = folded(inputs), partitioned(inputs), original(inputs)
            row["bf16_folded_vs_split"] = compare_tensors(baseline, output)
            row["bf16_original_vs_folded_split"] = compare_tensors(unchanged, output)
            assert list(output.shape) == [batch, 1000] and torch.isfinite(output).all()
            torch.save(output, args.output_dir / f"cpu-output-b{batch}.pt")
        prepared = load_model(bn_folded=True, output_split_channels=128)
        assert state_fingerprint(partitioned) == prepared._benchmark_transform["state_sha256"]
        result.update(status="ok", weights=weight_metadata(), model_transform=prepared._benchmark_transform,
                      fp32_assert_close={"rtol": 1e-4, "atol": 1e-4, "passed": True}, comparisons=comparisons)
        exit_code = 0
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        exit_code = 1
    write_json(args.output_dir / "validation.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
