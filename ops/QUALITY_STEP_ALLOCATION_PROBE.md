# Quality-tier step-allocation lever

## Scope

This branch adds an isolated request-time schedule lever for a controlled
quality-tier experiment. It does not change the schedule of requests that omit
the new fields.

- Current quality control: 16 stage-1 LTX2Scheduler transitions plus the
  official three-transition stage-2 schedule
  `[0.909375, 0.725, 0.421875, 0.0]`.
- Candidate: 15 stage-1 transitions plus four stage-2 transitions using
  `[0.909375, 0.77109375, 0.5734375, 0.31640625, 0.0]`.
- Both arms therefore use 19 total denoising transitions.

The hypothesis is narrow: spending one existing transition at full resolution
may improve terminal anatomy, duplicate-person suppression, and brightness
stability without adding model calls.

## Request contract

The candidate request adds:

```json
{
  "tier": "quality",
  "steps": 15,
  "stage2_sigmas": [
    0.909375,
    0.77109375,
    0.5734375,
    0.31640625,
    0.0
  ]
}
```

`stage2_sigmas` is accepted on both `fast` and `quality`. It is rejected
fail-closed unless it is:

- a JSON list containing 2–16 numeric points;
- finite, with every point in `[0.0, 1.0]`;
- strictly descending;
- terminated by exactly `0.0`.

`NaN`, positive/negative infinity, booleans, strings, duplicate points,
ascending segments, out-of-range values, and nonzero tails are rejected as
`invalid_request`.

An explicit `steps` value must be a JSON integer in `[1, 64]`. Omission retains
the existing tier default.

## Leakage and observability

The selected stage-2 tensor is carried in the request-local `settings` passed
to `run_case`. The handler no longer mutates
`stage_timing_runner.STAGE_2_DISTILLED_SIGMAS`, so one request cannot leave its
schedule behind for the next request.

`effective_config.schedule` attests:

- requested and effective stage-1 transition counts;
- requested and exact effective stage-2 sigma points;
- effective stage-2 transition count and schedule source;
- effective total transition count.

Any explicit schedule gets a config-tag suffix containing the stage counts and
a 12-hex SHA-256 prefix over the complete canonical sigma grid. Grids with the
same number of points but different values therefore produce different tags.

Compile-enabled workers also refresh a bounded `post_generation` snapshot after
every successful generation. It contains a monotonically increasing request
sequence, selected current cumulative Dynamo/Inductor counters, and current and
peak CUDA allocation/reservation. Only the newest generation is stored; no
per-request history grows in worker memory. Missing counters, CUDA, or warmed
compile state fail closed and suppress the generated artifact.

## Benchmark design

`run_quality_step_allocation_probe.js` defines a frozen 4-case × 2-seed ×
2-arm test (16 videos):

- samurai: terminal body/sword coherence;
- chef: extra-person and extra-limb entry;
- dancer: terminal darkening and body drift;
- waterfall: low-risk landscape motion control.

The runner uses the same accepted still for both arms of a case, freezes all
non-schedule controls, runs with concurrency one, and alternates `AB`/`BA`
within successive case-seed blocks. It validates exact response schedules,
runtime-tag separation, image hashes, canonical MP4 bytes, and ffprobe shape,
frame, FPS, codec, and audio evidence.

The default command is plan-only and makes zero network calls:

```bash
node ops/run_quality_step_allocation_probe.js
```

Endpoint execution requires the explicit `--execute` flag plus endpoint,
runtime-tag, OCI digest, still-directory, output-directory, and API-key
environment inputs. No endpoint was called while preparing this branch.
