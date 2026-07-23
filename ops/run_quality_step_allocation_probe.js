#!/usr/bin/env node
'use strict';

/**
 * Counterbalanced quality-tier schedule allocation benchmark.
 *
 * Default invocation is plan-only and performs no network calls:
 *   node ops/run_quality_step_allocation_probe.js
 *
 * Execution is deliberately opt-in and requires all provenance inputs:
 *   LTX_BENCHMARK_ENDPOINT_ID=... \
 *   LTX_BENCHMARK_EXPECT_TAG_PREFIX=... \
 *   LTX_BENCHMARK_IMAGE_DIGEST=sha256:... \
 *   LTX_BENCHMARK_STILLS_DIR=/absolute/path/to/accepted-stills \
 *   LTX_BENCHMARK_OUT=/absolute/path/to/new-output-directory \
 *   RUNPOD_MAIN_API_KEY=... \
 *   node ops/run_quality_step_allocation_probe.js --execute
 *
 * Accepted still filenames are `<case-id>__z_still.png`. The runner never
 * generates or substitutes stills, never overwrites an orphan artifact, runs
 * with concurrency one, and refuses outputs that do not attest the exact
 * requested/effective schedule.
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const SPEC_PATH = path.join(__dirname, 'quality_step_allocation_cases.json');
const POLL_MS = 2_000;
const TIMEOUT_MS = 30 * 60_000;
const MAX_CONCURRENCY = 1;
const EXECUTE = process.argv.includes('--execute');
const UNKNOWN_ARGS = process.argv.slice(2).filter((value) => value !== '--execute');

const QUALITY_DEFAULT_STAGE2 = [0.909375, 0.725, 0.421875, 0.0];
const CANDIDATE_STAGE2 = [
  0.909375,
  0.77109375,
  0.5734375,
  0.31640625,
  0.0,
];

const ARMS = [
  {
    id: 'quality_baseline_16_3',
    overrides: {},
    expectedSchedule: {
      stage1Requested: null,
      stage1Effective: 16,
      stage2Requested: null,
      stage2Effective: QUALITY_DEFAULT_STAGE2,
      stage2Steps: 3,
      totalTransitions: 19,
      source: 'quality_default',
    },
    hypothesis:
      'Current quality-tier control: 16 LTX2Scheduler transitions and the official three-transition stage-2 refiner.',
  },
  {
    id: 'quality_candidate_15_4',
    overrides: {
      steps: 15,
      stage2_sigmas: CANDIDATE_STAGE2,
    },
    expectedSchedule: {
      stage1Requested: 15,
      stage1Effective: 15,
      stage2Requested: CANDIDATE_STAGE2,
      stage2Effective: CANDIDATE_STAGE2,
      stage2Steps: 4,
      totalTransitions: 19,
      source: 'request',
    },
    hypothesis:
      'Move one of the same 19 total transitions from stage 1 into the full-resolution refiner to test terminal anatomy, duplicate-person, and brightness stability.',
  },
];

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

function fail(message) {
  throw new Error(message);
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function writeJsonAtomic(file, value) {
  const temporary = `${file}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, {
      mode: 0o600,
    });
    fs.renameSync(temporary, file);
    fs.chmodSync(file, 0o600);
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

function sha256Buffer(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function sha256File(file) {
  return sha256Buffer(fs.readFileSync(file));
}

function isSensitiveKey(key) {
  return /(?:^|_)(?:image|video|audio)?_?b64$|base64|token|secret|password|authorization|api[_-]?key/i.test(
    key,
  );
}

function stripSensitive(value, parentKey = '') {
  if (isSensitiveKey(parentKey)) {
    return typeof value === 'string'
      ? `<redacted ${value.length} chars>`
      : '<redacted>';
  }
  if (Array.isArray(value)) {
    return value.map((child) => stripSensitive(child, parentKey));
  }
  if (!value || typeof value !== 'object') {
    if (typeof value === 'string' && value.length > 1_000) {
      return `<redacted ${value.length} chars>`;
    }
    return value;
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, child]) => [
      key,
      stripSensitive(child, key),
    ]),
  );
}

function safeError(error) {
  return String(error?.message || error || 'unknown error')
    .replace(
      /(?:data:[^,;\s]+(?:;[^,\s]+)*,)?[A-Za-z0-9+/]{160,}={0,2}/g,
      '<redacted-large-blob>',
    )
    .slice(0, 1_500);
}

function exactArray(actual, expected) {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

function validateSpec(spec) {
  if (
    spec.schemaVersion !== 1 ||
    !spec.defaults ||
    spec.defaults.tier !== 'quality' ||
    !Array.isArray(spec.cases) ||
    spec.cases.length !== 4
  ) {
    fail('Spec must contain exactly four quality-tier cases');
  }
  const ids = new Set();
  for (const testCase of spec.cases) {
    if (
      typeof testCase.id !== 'string' ||
      !/^[a-z0-9_-]+$/.test(testCase.id) ||
      ids.has(testCase.id) ||
      typeof testCase.motionPrompt !== 'string' ||
      testCase.motionPrompt.length < 40 ||
      !Array.isArray(testCase.seeds) ||
      testCase.seeds.length !== 2 ||
      new Set(testCase.seeds).size !== 2 ||
      !testCase.seeds.every(
        (seed) => Number.isSafeInteger(seed) && seed >= 0,
      )
    ) {
      fail(`Malformed case ${JSON.stringify(stripSensitive(testCase))}`);
    }
    ids.add(testCase.id);
  }
}

function buildBlocks(spec) {
  const blocks = [];
  for (const testCase of spec.cases) {
    for (const seed of testCase.seeds) {
      blocks.push({ testCase, seed });
    }
  }
  return blocks;
}

function buildExecutionOrder(spec) {
  return buildBlocks(spec).flatMap((block, blockIndex) => {
    const armOrder = blockIndex % 2 === 0 ? ARMS : [...ARMS].reverse();
    return armOrder.map(
      (arm) => `${block.testCase.id}/seed-${block.seed}/${arm.id}`,
    );
  });
}

function printPlan(spec) {
  const executionOrder = buildExecutionOrder(spec);
  process.stdout.write(
    `${JSON.stringify(
      {
        mode: 'plan-only',
        networkCalls: 0,
        cases: spec.cases.length,
        seedsPerCase: 2,
        armsPerSeed: 2,
        totalJobs: executionOrder.length,
        maxConcurrency: MAX_CONCURRENCY,
        arms: ARMS,
        executionOrder,
      },
      null,
      2,
    )}\n`,
  );
}

function loadExecutionConfig() {
  const config = {
    endpointId: process.env.LTX_BENCHMARK_ENDPOINT_ID,
    expectedTagPrefix: process.env.LTX_BENCHMARK_EXPECT_TAG_PREFIX,
    imageDigest: process.env.LTX_BENCHMARK_IMAGE_DIGEST,
    stillsDirectory: process.env.LTX_BENCHMARK_STILLS_DIR,
    outputDirectory: process.env.LTX_BENCHMARK_OUT,
    apiKey: process.env.RUNPOD_MAIN_API_KEY,
  };
  for (const [key, value] of Object.entries(config)) {
    if (!value) fail(`Execution requires ${key}`);
  }
  if (!/^[a-zA-Z0-9]+$/.test(config.endpointId)) {
    fail('Endpoint id is malformed');
  }
  if (!/^sha256:[0-9a-f]{64}$/.test(config.imageDigest)) {
    fail('Image digest must be an exact sha256 OCI digest');
  }
  if (!path.isAbsolute(config.stillsDirectory)) {
    fail('Stills directory must be absolute');
  }
  if (!path.isAbsolute(config.outputDirectory)) {
    fail('Output directory must be absolute');
  }
  if (
    path.resolve(config.outputDirectory) === path.parse(config.outputDirectory).root
  ) {
    fail('Output directory cannot be a filesystem root');
  }
  return config;
}

async function providerApi(config, method, url, body) {
  const response = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  let parsed = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch (_) {
    parsed = null;
  }
  if (!response.ok) {
    fail(
      `${method} ${url} -> ${response.status}: ${JSON.stringify(
        stripSensitive(parsed || text),
      ).slice(0, 1_000)}`,
    );
  }
  return parsed;
}

function extractVideoBase64(output) {
  if (!output) return null;
  if (typeof output.video_b64 === 'string') return output.video_b64;
  if (output.output) return extractVideoBase64(output.output);
  return null;
}

function decodeCanonicalMp4(value, label) {
  if (
    typeof value !== 'string' ||
    value.length < 16 ||
    value.length % 4 !== 0 ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(value)
  ) {
    fail(`${label}: video_b64 is not canonical base64`);
  }
  const bytes = Buffer.from(value, 'base64');
  if (
    bytes.length < 12 ||
    bytes.subarray(4, 8).toString('ascii') !== 'ftyp' ||
    bytes.toString('base64') !== value
  ) {
    fail(`${label}: decoded payload is not a canonical ISO-BMFF MP4`);
  }
  return bytes;
}

function probeVideo(file, expected, label) {
  const result = spawnSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'format=format_name,size,duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames',
      '-of',
      'json',
      file,
    ],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    fail(`${label}: ffprobe failed: ${String(result.stderr).slice(0, 500)}`);
  }
  const probe = JSON.parse(result.stdout);
  const video = probe.streams?.find(
    (stream) => stream.codec_type === 'video',
  );
  const audio = probe.streams?.find(
    (stream) => stream.codec_type === 'audio',
  );
  const [numerator, denominator] = String(video?.r_frame_rate || '')
    .split('/')
    .map(Number);
  const measuredFps = numerator / denominator;
  if (
    !String(probe.format?.format_name || '').includes('mp4') ||
    video?.codec_name !== 'h264' ||
    video?.width !== expected.width ||
    video?.height !== expected.height ||
    Number(video?.nb_frames) !== expected.frames ||
    Math.abs(measuredFps - expected.fps) > 1e-6 ||
    (expected.audio === true && !audio)
  ) {
    fail(
      `${label}: MP4 attestation mismatch: ${JSON.stringify(
        stripSensitive(probe),
      ).slice(0, 1_000)}`,
    );
  }
  return probe;
}

function writeVideoAtomic(file, bytes, expected, label) {
  const temporary = `${file}.${process.pid}.part.mp4`;
  try {
    fs.writeFileSync(temporary, bytes, { mode: 0o600 });
    const probe = probeVideo(temporary, expected, label);
    fs.renameSync(temporary, file);
    fs.chmodSync(file, 0o600);
    return probe;
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

function assertEffectiveConfig(output, arm, input, config, label) {
  const effective = output?.effective_config;
  const schedule = effective?.schedule;
  const expected = arm.expectedSchedule;
  if (!effective || !schedule) fail(`${label}: missing effective schedule`);
  if (
    !String(output.config_tag || '').startsWith(
      `${config.expectedTagPrefix}-quality`,
    )
  ) {
    fail(`${label}: unexpected runtime config tag ${output.config_tag}`);
  }
  const hasExplicitScheduleTag = /-s1st\d+-s2st\d+-s2h[0-9a-f]{12}/.test(
    output.config_tag,
  );
  if (
    (arm.id === 'quality_candidate_15_4' && !hasExplicitScheduleTag) ||
    (arm.id === 'quality_baseline_16_3' && hasExplicitScheduleTag)
  ) {
    fail(`${label}: config tag does not distinguish the schedule arm`);
  }
  if (
    schedule.stage1?.requested_step_count !== expected.stage1Requested ||
    schedule.stage1?.effective_step_count !== expected.stage1Effective ||
    schedule.stage1?.explicit_sigma_grid !== false ||
    !(
      expected.stage2Requested === null
        ? schedule.stage2?.requested_sigmas === null
        : exactArray(
            schedule.stage2?.requested_sigmas,
            expected.stage2Requested,
          )
    ) ||
    !exactArray(
      schedule.stage2?.effective_sigmas,
      expected.stage2Effective,
    ) ||
    schedule.stage2?.effective_step_count !== expected.stage2Steps ||
    schedule.stage2?.source !== expected.source ||
    schedule.effective_total_transition_count !== expected.totalTransitions
  ) {
    fail(
      `${label}: schedule attestation mismatch: ${JSON.stringify(
        stripSensitive(schedule),
      )}`,
    );
  }
  if (
    effective.sampler?.stage1 !== 'gradient_estimating_euler' ||
    effective.sampler?.stage1_ge !== true ||
    effective.sampler?.stage2 !== 'euler' ||
    effective.tier !== 'quality' ||
    effective.audio?.enabled !== true ||
    effective.enhance?.requested !== false ||
    effective.enhance?.applied !== false ||
    effective.cas?.amount !== 0 ||
    effective.cas?.enabled !== false ||
    effective.cfg_cache?.enabled !== true ||
    effective.cfg_cache?.range?.start !== 4 ||
    effective.cfg_cache?.range?.end_exclusive !== 9
  ) {
    fail(
      `${label}: frozen non-schedule config mismatch: ${JSON.stringify(
        stripSensitive(effective),
      ).slice(0, 1_500)}`,
    );
  }
  for (const key of ['width', 'height', 'frames', 'fps']) {
    if (effective.shape?.[key] !== input[key]) {
      fail(`${label}: effective shape.${key} differs from input`);
    }
  }
  return effective;
}

async function runJob(config, input, arm, label) {
  const endpoint = `https://api.runpod.ai/v2/${config.endpointId}`;
  const wallStart = Date.now();
  const submitted = await providerApi(config, 'POST', `${endpoint}/run`, {
    input,
  });
  if (!submitted?.id) fail(`${label}: provider returned no job id`);
  try {
    let terminal = null;
    while (Date.now() - wallStart < TIMEOUT_MS) {
      await sleep(POLL_MS);
      const status = await providerApi(
        config,
        'GET',
        `${endpoint}/status/${submitted.id}`,
      );
      if (
        ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'].includes(
          status?.status,
        )
      ) {
        terminal = status;
        break;
      }
    }
    if (!terminal) fail(`${label}: client timeout`);
    if (terminal.status !== 'COMPLETED') {
      fail(
        `${label}: provider ${terminal.status}: ${JSON.stringify(
          stripSensitive(terminal.error || terminal.output),
        ).slice(0, 1_000)}`,
      );
    }
    const effectiveConfig = assertEffectiveConfig(
      terminal.output,
      arm,
      input,
      config,
      label,
    );
    return {
      providerJobId: submitted.id,
      startedAt: new Date(wallStart).toISOString(),
      completedAt: new Date().toISOString(),
      wallMs: Date.now() - wallStart,
      delayMs: terminal.delayTime ?? null,
      executionMs: terminal.executionTime ?? null,
      output: terminal.output,
      effectiveConfig,
    };
  } catch (error) {
    error.providerJobId = submitted.id;
    throw error;
  }
}

function collectStills(spec, config) {
  return Object.fromEntries(
    spec.cases.map((testCase) => {
      const file = path.join(
        config.stillsDirectory,
        `${testCase.id}__z_still.png`,
      );
      if (!fs.existsSync(file)) fail(`Missing accepted still ${file}`);
      return [
        testCase.id,
        {
          file,
          bytes: fs.statSync(file).size,
          sha256: sha256File(file),
        },
      ];
    }),
  );
}

function resultKey(testCase, seed, arm) {
  return `${testCase.id}/seed-${seed}/${arm.id}`;
}

async function executeBenchmark(spec) {
  const config = loadExecutionConfig();
  const stills = collectStills(spec, config);
  const executionOrder = buildExecutionOrder(spec);
  const runnerSha256 = sha256File(__filename);
  const specSha256 = sha256File(SPEC_PATH);
  fs.mkdirSync(config.outputDirectory, { recursive: true });
  const manifestPath = path.join(config.outputDirectory, 'manifest.json');
  const manifest = fs.existsSync(manifestPath)
    ? readJson(manifestPath)
    : {
        schemaVersion: 1,
        status: 'running',
        startedAt: new Date().toISOString(),
        purpose:
          'Counterbalanced same-still quality-tier 16+3 versus 15+4 allocation benchmark at the same 19 transitions.',
        endpointId: config.endpointId,
        expectedRuntimeTagPrefix: config.expectedTagPrefix,
        imageDigest: config.imageDigest,
        harness: {
          runner: path.relative(ROOT, __filename),
          runnerSha256,
          spec: path.relative(ROOT, SPEC_PATH),
          specSha256,
          node: process.version,
          platform: `${process.platform}/${process.arch}`,
        },
        arms: ARMS,
        sourceStills: stills,
        executionOrder,
        maxConcurrency: MAX_CONCURRENCY,
        results: [],
        failures: [],
      };
  if (
    manifest.schemaVersion !== 1 ||
    manifest.endpointId !== config.endpointId ||
    manifest.expectedRuntimeTagPrefix !== config.expectedTagPrefix ||
    manifest.imageDigest !== config.imageDigest ||
    manifest.harness?.runnerSha256 !== runnerSha256 ||
    manifest.harness?.specSha256 !== specSha256 ||
    JSON.stringify(manifest.arms) !== JSON.stringify(ARMS) ||
    JSON.stringify(manifest.sourceStills) !== JSON.stringify(stills) ||
    JSON.stringify(manifest.executionOrder) !== JSON.stringify(executionOrder)
  ) {
    fail(
      'Refusing to mix endpoint, image, runtime tag, runner, spec, still, arm, or order provenance',
    );
  }
  const expectedKeys = new Set(executionOrder);
  const completedKeys = new Set();
  for (const result of manifest.results) {
    const key = `${result.caseId}/seed-${result.seed}/${result.arm}`;
    const artifactPath = path.resolve(
      config.outputDirectory,
      result.artifact?.file || '',
    );
    if (
      !expectedKeys.has(key) ||
      completedKeys.has(key) ||
      !artifactPath.startsWith(
        `${path.resolve(config.outputDirectory)}${path.sep}`,
      ) ||
      !fs.existsSync(artifactPath) ||
      sha256File(artifactPath) !== result.artifact?.sha256 ||
      result.imageDigest !== config.imageDigest ||
      result.inputImageSha256 !== stills[result.caseId]?.sha256
    ) {
      fail(`Resume integrity failed for ${key}`);
    }
    completedKeys.add(key);
  }
  manifest.status = 'running';
  delete manifest.completedAt;
  writeJsonAtomic(manifestPath, manifest);

  const blocks = buildBlocks(spec);
  for (let blockIndex = 0; blockIndex < blocks.length; blockIndex += 1) {
    const { testCase, seed } = blocks[blockIndex];
    const armOrder = blockIndex % 2 === 0 ? ARMS : [...ARMS].reverse();
    const image = fs
      .readFileSync(stills[testCase.id].file)
      .toString('base64');
    for (const arm of armOrder) {
      const key = resultKey(testCase, seed, arm);
      if (completedKeys.has(key)) continue;
      const outputFile = path.join(
        config.outputDirectory,
        `${testCase.id}__seed-${seed}__${arm.id}.mp4`,
      );
      if (fs.existsSync(outputFile)) {
        fail(`Refusing to overwrite orphan artifact ${outputFile}`);
      }
      const input = {
        prompt: testCase.motionPrompt,
        ...spec.defaults,
        seed,
        image_b64: image,
        ...arm.overrides,
      };
      let providerJobId = null;
      try {
        const result = await runJob(config, input, arm, key);
        providerJobId = result.providerJobId;
        const encodedVideo = extractVideoBase64(result.output);
        if (!encodedVideo) fail(`${key}: missing video_b64`);
        const videoBytes = decodeCanonicalMp4(encodedVideo, key);
        const probe = writeVideoAtomic(
          outputFile,
          videoBytes,
          input,
          key,
        );
        const outputMetadata = stripSensitive(result.output);
        delete result.output;
        manifest.results.push({
          caseId: testCase.id,
          expectedPeople: testCase.expectedPeople,
          seed,
          arm: arm.id,
          imageDigest: config.imageDigest,
          inputImageSha256: stills[testCase.id].sha256,
          input: stripSensitive(input),
          artifact: {
            file: path.relative(config.outputDirectory, outputFile),
            bytes: videoBytes.length,
            sha256: sha256Buffer(videoBytes),
            probe,
          },
          ...result,
          outputMetadata,
        });
        completedKeys.add(key);
        writeJsonAtomic(manifestPath, manifest);
        process.stdout.write(
          `${key} completed in ${(result.wallMs / 1_000).toFixed(1)}s\n`,
        );
      } catch (error) {
        manifest.failures.push({
          caseId: testCase.id,
          seed,
          arm: arm.id,
          failedAt: new Date().toISOString(),
          providerJobId: error.providerJobId || providerJobId,
          imageDigest: config.imageDigest,
          input: stripSensitive(input),
          inputImageSha256: stills[testCase.id].sha256,
          error: safeError(error),
        });
        manifest.status = 'failed';
        writeJsonAtomic(manifestPath, manifest);
        throw error;
      }
    }
  }

  if (
    manifest.results.length !== expectedKeys.size ||
    [...expectedKeys].some((key) => !completedKeys.has(key))
  ) {
    fail('Benchmark did not complete the exact expected 16-result set');
  }
  for (const { testCase, seed } of blocks) {
    const pair = manifest.results.filter(
      (result) => result.caseId === testCase.id && result.seed === seed,
    );
    if (
      pair.length !== 2 ||
      new Set(pair.map((result) => result.outputMetadata.config_tag)).size !==
        2
    ) {
      fail(
        `${testCase.id}/seed-${seed}: schedule arms lack distinct runtime tags`,
      );
    }
  }
  manifest.status = 'completed';
  manifest.completedAt = new Date().toISOString();
  writeJsonAtomic(manifestPath, manifest);
  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        outputDirectory: config.outputDirectory,
        results: manifest.results.length,
      },
      null,
      2,
    )}\n`,
  );
}

async function main() {
  if (UNKNOWN_ARGS.length) {
    fail(`Unknown arguments: ${UNKNOWN_ARGS.join(', ')}`);
  }
  const spec = readJson(SPEC_PATH);
  validateSpec(spec);
  if (!EXECUTE) {
    printPlan(spec);
    return;
  }
  await executeBenchmark(spec);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${safeError(error)}\n`);
    process.exit(1);
  });
}

module.exports = {
  ARMS,
  CANDIDATE_STAGE2,
  QUALITY_DEFAULT_STAGE2,
  assertEffectiveConfig,
  buildExecutionOrder,
  decodeCanonicalMp4,
  validateSpec,
};
