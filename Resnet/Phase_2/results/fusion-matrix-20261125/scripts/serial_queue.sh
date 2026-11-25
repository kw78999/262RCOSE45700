#!/bin/bash
# Serial finish of the fusion matrix (user request 2026-11-25: at most one job at a time).
# Re-runnable: every step skips itself when its result already exists. Events -> queue.jsonl.
cd /root
R=Resnet/Phase_2/results; M=$R/fusion-matrix-20261125; F4=$R/matrix-f4-20261125
export XDG_CACHE_HOME=/root/Resnet/.cache TMPDIR=/root/Resnet/.tmp
event(){ echo "{\"step\": \"$1\", \"event\": \"$2\", \"t\": $(date +%s), \"kst\": \"$(TZ=Asia/Seoul date +%H:%M:%S)\"${3:+, \"rc\": $3}}" >> $M/queue.jsonl; echo "$(TZ=Asia/Seoul date +%H:%M:%S) $1 $2 ${3:-}"; }
finished(){ grep -q '"status": "finished"' $R/$1/manifest.json 2>/dev/null; }
resume(){  # $1 step name, $2 run dir
  if finished $2; then event $1 skip; return; fi
  event $1 start
  /root/venv/bin/python -u -m Resnet.Phase_2.run --resume --run-dir $R/$2 >> $M/runner-${2%-20261125}.log 2>&1
  event $1 end $?
}

/root/venv/bin/python -c "from Resnet.Phase_2.run import toolchain_preflight as t; import sys; sys.exit(0 if t()['ok'] else 1)" \
  || { event preflight failed 1; exit 1; }

# 1. f4 eager-b1 re-run (its first timed loops overlapped the compile probes)
J=$F4/jobs
if [ -d $J/eager-b1-t1.contended ] && [ -f $J/eager-b1-t1/result.json ]; then event f4-eager-b1-rerun skip
else
  event f4-eager-b1-rerun start
  [ -d $J/eager-b1-t1.contended ] || { mv $J/eager-b1-t1 $J/eager-b1-t1.contended; mv $F4/caches/eager-b1-t1 $F4/caches/eager-b1-t1.contended; }
  rm -rf $J/eager-b1-t1 $F4/caches/eager-b1-t1   # only a half-finished re-run can be here
  mkdir -p $F4/tmp/eager-b1-t1-rerun
  (cd $F4/source && env FURIOSA_TORCH_LOG=error FURIOSA_TORCH_PROFILE=error OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     TMPDIR=/root/$F4/tmp/eager-b1-t1-rerun /root/venv/bin/python -u -m Resnet.Phase_2.worker \
     --config /root/$F4/config.json --mode eager --batch 1 --trial 1 \
     --cache /root/$F4/caches/eager-b1-t1 --output /root/$J/eager-b1-t1 > /root/$J/eager-b1-t1.rerun.log 2>&1)
  event f4-eager-b1-rerun end $?
fi

# 2-4. f2 eager-b16 | f1 eager-b64 | f1 compile-b64 (moved jobs exit at once as pointers)
resume f2-eager-b16 matrix-f2-20261125
resume f1-eager-b64 matrix-f1-20261125
resume f1-compile-b64 matrix-f1x-20261125

# 5. f2 b64 op probe -> b64 Eager plan for the f2x run
P=$R/eager-op-probe-fusion2-b64
if [ "$(($(wc -l < $P/probe.csv) - 1))" -ge 44 ]; then event f2-b64-probe skip
else
  event f2-b64-probe start
  /root/venv/bin/python -u -m Resnet.Phase_2.eager_op_probe --output $P --num-fusion 2 --device-index 2 --batches 64 >> $P/probe-serial.log 2>&1
  event f2-b64-probe end $?
fi
/root/venv/bin/python -m Resnet.Phase_2.eager_fix_plan --probe $R/eager-op-probe-fusion2/probe.csv $P/probe.csv \
  --batches 16 64 --chunk 1 --output Resnet/Phase_2/eager_batch_partition_fusion2.json 2>&1 | grep -v -i warn \
  || { event f2-b64-plan failed 1; exit 1; }
/root/venv/bin/python - <<'PY' || { event f2-b64-plan failed 1; exit 1; }
import json
from pathlib import Path
plan = json.load(open("Resnet/Phase_2/eager_batch_partition_fusion2.json"))["batches"]
f2 = json.loads(Path("Resnet/Phase_2/results/matrix-f2-20261125/config.json").read_text())
assert f2["eager_batch_partition"]["16"]["modules"] == plan["16"]["modules"], "b16 plan changed"
p = Path("Resnet/Phase_2/results/matrix-f2x-20261125/config.json"); c = json.loads(p.read_text())
c["eager_batch_partition"]["64"] = {"chunk": plan["64"]["chunk"], "modules": plan["64"]["modules"]}
p.write_text(json.dumps(c, indent=2) + "\n"); print("f2x b64 plan:", len(plan["64"]["modules"]), "modules")
PY
event f2-b64-plan done

# 6-7. f2 compile-b64, then f2 eager-b64
resume f2-compile-eager-b64 matrix-f2x-20261125

event all done
