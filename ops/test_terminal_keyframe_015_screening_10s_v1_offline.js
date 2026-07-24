#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const here = __dirname;
const runner = path.join(here, 'run_vbvr_screening_10s_v1.js');
const specPath = path.join(
  here,
  'terminal_keyframe_015_screening_10s_v1.json',
);
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));

assert.strictEqual(
  spec.experimentId,
  'terminal-keyframe-stage1-015-screening-10s-v1',
);
assert.strictEqual(spec.cases.length, 4);
assert.strictEqual(spec.defaults.frames, 241);
assert.strictEqual(spec.defaults.terminal_keyframe_strength_stage1, 0.15);
assert.strictEqual(spec.defaults.terminal_keyframe_strength_stage2, 0);

const result = spawnSync(process.execPath, [runner], {
  encoding: 'utf8',
  env: {
    PATH: process.env.PATH,
    LTX_SCREEN_SPEC_PATH: specPath,
  },
});
assert.strictEqual(result.status, 0, result.stderr);
const plan = JSON.parse(result.stdout);
assert.strictEqual(plan.execute, false);
assert.strictEqual(plan.externalRequests, 0);
assert.strictEqual(plan.posts, 5);
assert.strictEqual(plan.scored, 4);
assert.strictEqual(plan.playbackMuted, true);

process.stdout.write(
  `${JSON.stringify({
    ok: true,
    offline: true,
    externalRequests: 0,
    posts: plan.posts,
    scored: plan.scored,
  })}\n`,
);
