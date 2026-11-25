2026-11-25 11:06 KST: after eager-b1-t1, `source/Resnet/Phase_2/worker.py` and `config.json` were patched
(`pause_compile_probes: true`): the worker SIGSTOPs eager_op_probe process groups right after entering
the warmup stage and SIGCONTs them after the aux loop (and in `finally`). Reason: the 0.5 s polling
gate (probe_gate.py) reacted too late, so eager-b1-t1's warmup and repeats ran next to the probes
(repeat SD 2.47 ms). eager-b1-t1 is re-run at the end on a fresh cache; the contended copy is kept
as jobs/eager-b1-t1.contended. source_hashes.json predates this patch.

2026-11-25 12:10 KST: second patch (user asked to use PE 0-3 in parallel). From compile-b64-t1 on, the
worker uses `exclusive.py`: a host-wide lock around warmup/repeat/aux, a wait until no other job is in
a timed stage, and SIGSTOP of every other Resnet runner/worker/probe for that section (intervals in
fusion-matrix-20261125/pauses.jsonl). `first_call_wall` timestamps allow subtracting those pauses.
config.json: `pause_compile_probes` replaced by `exclusive_measurement`. eager-b64-t1 still ran the
first patch (probes paused, other workers protected only by the 0.2 s gate).
