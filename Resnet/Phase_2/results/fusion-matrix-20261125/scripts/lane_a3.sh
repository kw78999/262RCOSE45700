#!/bin/bash
# Lane A after the 14:34 KST process loss: f2 b64 probe (PE 4-5) -> f2 b64 plan into the f2 run -> after f2x, f4 eager-b1 re-run (PE 4-7).
cd /root; R=Resnet/Phase_2/results; F4=$R/matrix-f4-20261125; F2=$R/matrix-f2-20261125
while kill -0 3520 2>/dev/null; do sleep 10; done
echo "$(date -u +%T) f2 b64 probe exited"
if [ -d $F2/jobs/eager-b64-t1 ]; then echo "WARNING: f2 eager-b64 started before its plan"; fi
XDG_CACHE_HOME=/root/Resnet/.cache /root/venv/bin/python -m Resnet.Phase_2.eager_fix_plan --probe $R/eager-op-probe-fusion2/probe.csv $R/eager-op-probe-fusion2-b64/probe.csv --batches 16 64 --chunk 1 --output Resnet/Phase_2/eager_batch_partition_fusion2.json 2>&1 | grep -v -i warn
/root/venv/bin/python - <<'PY'
import json
from pathlib import Path
plan=json.load(open("Resnet/Phase_2/eager_batch_partition_fusion2.json"))["batches"]
for path in [Path("Resnet/Phase_2/config-matrix-f2.json"), Path("Resnet/Phase_2/results/matrix-f2-20261125/config.json")]:
    c=json.loads(path.read_text())
    assert c["eager_batch_partition"]["16"]["modules"] == plan["16"]["modules"], "b16 plan changed"
    c["eager_batch_partition"]["64"]={"chunk": plan["64"]["chunk"], "modules": plan["64"]["modules"]}
    path.write_text(json.dumps(c, indent=2)+"\n"); print("b64 plan ->", path)
PY
until grep -q '"status": "finished"' $R/matrix-f2x-20261125/manifest.json 2>/dev/null; do sleep 15; done
echo "$(date -u +%T) f2x finished; f4 eager-b1 re-run"
J=$F4/jobs; [ -d $J/eager-b1-t1.contended ] || { mv $J/eager-b1-t1 $J/eager-b1-t1.contended; mv $F4/caches/eager-b1-t1 $F4/caches/eager-b1-t1.contended; }
mkdir -p $F4/tmp/eager-b1-t1-rerun
(cd $F4/source && env FURIOSA_TORCH_LOG=error FURIOSA_TORCH_PROFILE=error OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XDG_CACHE_HOME=/root/Resnet/.cache TMPDIR=/root/$F4/tmp/eager-b1-t1-rerun \
  setsid /root/venv/bin/python -u -m Resnet.Phase_2.worker --config /root/$F4/config.json --mode eager --batch 1 --trial 1 \
  --cache /root/$F4/caches/eager-b1-t1 --output /root/$J/eager-b1-t1 > /root/$J/eager-b1-t1.rerun.log 2>&1)
echo "$(date -u +%T) f4 eager-b1 re-run exited $?"
