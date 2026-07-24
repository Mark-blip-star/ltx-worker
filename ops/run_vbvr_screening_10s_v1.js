#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const SPEC_PATH = process.env.LTX_SCREEN_SPEC_PATH
  ? path.resolve(process.env.LTX_SCREEN_SPEC_PATH)
  : path.join(__dirname, 'vbvr_screening_10s_v1.json');
const EXECUTE = process.argv.includes('--execute');
const POLL_MS = 2_000;
const JOB_TIMEOUT_MS = 45 * 60_000;
const MAX_POSTS = 5;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (value) =>
  crypto.createHash('sha256').update(value).digest('hex');

function fail(message) {
  throw new Error(message);
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function safeError(error, apiKey = '') {
  return String(error?.message || error || 'unknown error')
    .split(apiKey)
    .join('<RUNPOD_MAIN_API_KEY>')
    .replace(
      /(?:data:[^,;\s]+(?:;[^,\s]+)*,)?[A-Za-z0-9+/]{160,}={0,2}/g,
      '<redacted-large-blob>',
    )
    .slice(0, 2_000);
}

function stripLarge(value, parentKey = '') {
  if (
    /(?:^|_)(?:image|video|audio)?_?b64$|base64|token|secret|password|authorization|api[_-]?key/i.test(
      parentKey,
    )
  ) {
    return '<redacted>';
  }
  if (Array.isArray(value)) {
    return value.map((item) => stripLarge(item, parentKey));
  }
  if (!value || typeof value !== 'object') {
    if (typeof value === 'string' && value.length > 1_000) {
      return `<${value.length} chars>`;
    }
    return value;
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, child]) => [
      key,
      stripLarge(child, key),
    ]),
  );
}

function writeJsonAtomic(file, value) {
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    mode: 0o600,
  });
  fs.renameSync(temporary, file);
  fs.chmodSync(file, 0o600);
}

async function api(apiKey, method, url, body) {
  const response = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    redirect: 'error',
  });
  const text = await response.text();
  let parsed = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch (_) {
    parsed = null;
  }
  if (!response.ok) {
    const error = new Error(
      `${method} ${url} failed (${response.status}): ${safeError(
        parsed ? JSON.stringify(stripLarge(parsed)) : text,
        apiKey,
      )}`,
    );
    error.httpStatus = response.status;
    error.providerCode = parsed?.code;
    throw error;
  }
  return parsed;
}

async function runJob(apiKey, endpointId, input) {
  const base = `https://api.runpod.ai/v2/${endpointId}`;
  const startedAt = Date.now();
  let submitted = null;
  while (!submitted && Date.now() - startedAt < 20 * 60_000) {
    try {
      submitted = await api(apiKey, 'POST', `${base}/run`, { input });
    } catch (error) {
      if (
        error.httpStatus === 409 &&
        ['ENDPOINT_PAUSED', 'INITIALIZING'].includes(error.providerCode)
      ) {
        await sleep(POLL_MS);
        continue;
      }
      throw error;
    }
  }
  if (!submitted?.id) fail('RunPod did not return a durable job id');
  while (Date.now() - startedAt < JOB_TIMEOUT_MS) {
    await sleep(POLL_MS);
    const status = await api(
      apiKey,
      'GET',
      `${base}/status/${submitted.id}`,
    );
    if (
      ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'].includes(
        status.status,
      )
    ) {
      return {
        providerJobId: submitted.id,
        status: status.status,
        wallMs: Date.now() - startedAt,
        delayMs: status.delayTime,
        executionMs: status.executionTime,
        output: status.output,
        error: status.error,
      };
    }
  }
  fail(`Job ${submitted.id} exceeded the client timeout`);
}

function extractVideoBase64(output) {
  if (!output) return null;
  if (typeof output === 'string' && output.length > 1_000) return output;
  if (typeof output.video_b64 === 'string') return output.video_b64;
  if (output.output) return extractVideoBase64(output.output);
  if (Array.isArray(output.videos) && output.videos[0]) {
    return extractVideoBase64(output.videos[0]);
  }
  return null;
}

function decodeCanonicalMp4(value) {
  if (
    typeof value !== 'string' ||
    value.length < 16 ||
    value.length % 4 !== 0 ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(value)
  ) {
    fail('Worker video is not canonical base64');
  }
  const bytes = Buffer.from(value, 'base64');
  if (
    bytes.length < 12 ||
    bytes.subarray(4, 8).toString('ascii') !== 'ftyp' ||
    bytes.toString('base64') !== value
  ) {
    fail('Worker output is not a canonical ISO-BMFF MP4');
  }
  return bytes;
}

function probeVideo(file, defaults) {
  const result = spawnSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-show_entries',
      'stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames',
      '-show_entries',
      'format=duration',
      '-of',
      'json',
      file,
    ],
    { encoding: 'utf8' },
  );
  if (result.status !== 0) {
    fail(`ffprobe failed for ${path.basename(file)}`);
  }
  const parsed = JSON.parse(result.stdout);
  const video = parsed.streams.find((stream) => stream.codec_type === 'video');
  const audio = parsed.streams.find((stream) => stream.codec_type === 'audio');
  const duration = Number(parsed.format?.duration);
  if (
    video?.codec_name !== 'h264' ||
    Number(video.width) !== defaults.width ||
    Number(video.height) !== defaults.height ||
    Number(video.nb_frames) !== defaults.frames ||
    video.r_frame_rate !== `${defaults.fps}/1` ||
    audio?.codec_name !== 'aac' ||
    Math.abs(duration - defaults.frames / defaults.fps) > 0.03
  ) {
    fail(`Canonical media contract failed for ${path.basename(file)}`);
  }
  return {
    videoCodec: video.codec_name,
    audioCodec: audio.codec_name,
    width: Number(video.width),
    height: Number(video.height),
    frames: Number(video.nb_frames),
    fps: defaults.fps,
    duration,
  };
}

function assertOutput(output, expectedTag, defaults) {
  const effective = output?.effective_config || {};
  const shape = effective.shape || {};
  const imageConditioning = effective.image_conditioning || {};
  const terminalKeyframe = effective.terminal_keyframe || {};
  const expectsTerminalKeyframe =
    Number(defaults.terminal_keyframe_strength_stage1 || 0) > 0 ||
    Number(defaults.terminal_keyframe_strength_stage2 || 0) > 0;
  if (
    output?.config_tag !== expectedTag ||
    effective.tier !== 'quality' ||
    effective.sampler?.stage1 !== 'gradient_estimating_euler' ||
    effective.sampler?.stage1_ge !== true ||
    effective.sampler?.stage2 !== 'euler' ||
    effective.cfg_cache?.enabled !== true ||
    effective.cfg_cache?.configured_range !== '4:9' ||
    effective.cas?.enabled !== false ||
    effective.enhance?.requested !== false ||
    effective.enhance?.applied !== false ||
    effective.audio?.enabled !== true ||
    shape.width !== defaults.width ||
    shape.height !== defaults.height ||
    shape.frames !== defaults.frames ||
    shape.fps !== defaults.fps ||
    imageConditioning.enabled !== true ||
    imageConditioning.source !== 'request_stage_pair' ||
    imageConditioning.frame_index !== 0 ||
    imageConditioning.strength !== 0.8 ||
    imageConditioning.stage1_strength !== 0.8 ||
    imageConditioning.stage2_strength !== 0.8 ||
    effective.schedule?.stage1?.effective_step_count !== 16 ||
    effective.schedule?.stage2?.effective_step_count !== 3 ||
    effective.schedule?.effective_total_transition_count !== 19 ||
    (expectsTerminalKeyframe &&
      (terminalKeyframe.enabled !== true ||
        terminalKeyframe.frame_index !== defaults.frames - 1 ||
        terminalKeyframe.stage1_strength !==
          Number(defaults.terminal_keyframe_strength_stage1 || 0) ||
        terminalKeyframe.stage2_strength !==
          Number(defaults.terminal_keyframe_strength_stage2 || 0) ||
        terminalKeyframe.source !== 'request'))
  ) {
    fail(
      `Effective runtime attestation failed: ${JSON.stringify(
        stripLarge(output),
      ).slice(0, 2_000)}`,
    );
  }
}

function validateSpec(spec) {
  if (
    spec.schemaVersion !== 1 ||
    typeof spec.experimentId !== 'string' ||
    !spec.experimentId ||
    spec.cases?.length !== 4 ||
    spec.defaults?.frames !== 241 ||
    spec.defaults?.audio !== true ||
    typeof spec.artifactSuffix !== 'string' ||
    !spec.artifactSuffix
  ) {
    fail('Four-case 10s screening spec contract is invalid');
  }
}

function loadFreeze(spec) {
  const freezePath = process.env.LTX_VBVR_SCREEN_FREEZE_PATH;
  const expectedSha = process.env.LTX_VBVR_SCREEN_FREEZE_SHA256;
  if (!freezePath || !/^[0-9a-f]{64}$/.test(expectedSha || '')) {
    fail('Exact freeze path and SHA-256 are required for execution');
  }
  const bytes = fs.readFileSync(path.resolve(freezePath));
  if (sha256(bytes) !== expectedSha) fail('VBVR freeze SHA-256 mismatch');
  const freeze = JSON.parse(bytes.toString('utf8'));
  if (
    freeze.schemaVersion !== 1 ||
    freeze.status !== 'FROZEN_FOR_EXECUTION' ||
    freeze.experimentId !== spec.experimentId ||
    !/^[0-9a-f]{64}$/.test(freeze.image?.ociDigestHex || '') ||
    (spec.requireExtraLora === true &&
      (!/^[0-9a-f]{64}$/.test(freeze.extraLora?.sha256 || '') ||
        !/^[0-9a-f]{40}$/.test(freeze.extraLora?.revision || ''))) ||
    freeze.endpoint?.workersMin !== 1 ||
    freeze.endpoint?.workersMax !== 1 ||
    freeze.endpoint?.idleTimeout !== 600 ||
    freeze.endpoint?.flashboot !== false
  ) {
    fail('VBVR freeze contents are incomplete or unsafe');
  }
  return freeze;
}

async function execute(spec) {
  const apiKey = process.env.RUNPOD_MAIN_API_KEY;
  const outDir = path.resolve(process.env.LTX_VBVR_SCREEN_OUT || '');
  const stillsDir = path.resolve(process.env.LTX_VBVR_SCREEN_STILLS_DIR || '');
  if (!apiKey || !outDir || !stillsDir) {
    fail('Execution requires runtime API key, output directory, and stills');
  }
  const freeze = loadFreeze(spec);
  const endpoint = await api(
    apiKey,
    'GET',
    `https://rest.runpod.io/v1/endpoints/${freeze.endpoint.id}`,
  );
  const observed = {
    id: endpoint.id,
    templateId: endpoint.templateId,
    workersMin: endpoint.workersMin,
    workersMax: endpoint.workersMax,
    idleTimeout: endpoint.idleTimeout,
    flashboot: endpoint.flashboot,
  };
  if (JSON.stringify(observed) !== JSON.stringify(freeze.endpoint)) {
    fail('Live endpoint differs from the frozen fixed-pool policy');
  }

  fs.mkdirSync(outDir, { recursive: false });
  const manifestPath = path.join(outDir, 'manifest.json');
  const jobs = [
    {
      caseId: spec.warmup.caseId,
      seed: spec.warmup.seed,
      scored: false,
    },
    ...spec.cases.map((testCase) => ({
      caseId: testCase.id,
      seed: testCase.seed,
      scored: true,
    })),
  ];
  if (jobs.length !== MAX_POSTS) fail('Unexpected VBVR POST count');
  const manifest = {
    schemaVersion: 1,
    experimentId: spec.experimentId,
    status: 'running',
    startedAt: new Date().toISOString(),
    freeze,
    endpointAttestation: observed,
    playbackPolicy: {
      muted: true,
      volume: 0,
      autoplay: false,
      audioQualityScored: false,
    },
    results: [],
    secretsStored: false,
  };
  writeJsonAtomic(manifestPath, manifest);

  for (const job of jobs) {
    const testCase = spec.cases.find((item) => item.id === job.caseId);
    const stillPath = path.join(stillsDir, `${job.caseId}__z_still.png`);
    const image = fs.readFileSync(stillPath);
    if (sha256(image) !== freeze.stills[job.caseId].sha256) {
      fail(`${job.caseId}: still SHA-256 mismatch`);
    }
    const input = {
      ...spec.defaults,
      prompt: testCase.motionPrompt,
      image_b64: image.toString('base64'),
      seed: job.seed,
    };
    const result = await runJob(apiKey, freeze.endpoint.id, input);
    if (result.status !== 'COMPLETED') {
      fail(`${job.caseId}: terminal status ${result.status}`);
    }
    assertOutput(result.output, freeze.expectedConfigTag, spec.defaults);
    const video = decodeCanonicalMp4(extractVideoBase64(result.output));
    const fileName = job.scored
      ? `${job.caseId}__seed-${job.seed}__${spec.artifactSuffix}.mp4`
      : `__warmup__${job.caseId}__seed-${job.seed}__${spec.artifactSuffix}.mp4`;
    const filePath = path.join(outDir, fileName);
    fs.writeFileSync(filePath, video, { mode: 0o600, flag: 'wx' });
    const probe = probeVideo(filePath, spec.defaults);
    manifest.results.push({
      caseId: job.caseId,
      seed: job.seed,
      scored: job.scored,
      providerJobId: result.providerJobId,
      wallMs: result.wallMs,
      delayMs: result.delayMs,
      executionMs: result.executionMs,
      artifactFile: fileName,
      artifactBytes: video.length,
      artifactSha256: sha256(video),
      promptSha256: sha256(testCase.motionPrompt),
      stillSha256: sha256(image),
      configTag: result.output.config_tag,
      timers: stripLarge(result.output.timers),
      effectiveConfig: stripLarge(result.output.effective_config),
      probe,
    });
    writeJsonAtomic(manifestPath, manifest);
    process.stdout.write(`${job.caseId} completed (${result.executionMs} ms)\n`);
  }

  manifest.status = 'completed';
  manifest.completedAt = new Date().toISOString();
  writeJsonAtomic(manifestPath, manifest);
  process.stdout.write(
    `${JSON.stringify({
      ok: true,
      outDir,
      results: manifest.results.length,
      scored: manifest.results.filter((item) => item.scored).length,
    })}\n`,
  );
}

async function main() {
  const unknown = process.argv.slice(2).filter((arg) => arg !== '--execute');
  if (unknown.length) fail(`Unknown arguments: ${unknown.join(', ')}`);
  const spec = readJson(SPEC_PATH);
  validateSpec(spec);
  if (!EXECUTE) {
    process.stdout.write(
      `${JSON.stringify({
        ok: true,
        execute: false,
        externalRequests: 0,
        experimentId: spec.experimentId,
        posts: MAX_POSTS,
        scored: spec.cases.length,
        audio: spec.defaults.audio,
        playbackMuted: true,
      })}\n`,
    );
    return;
  }
  await execute(spec);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${safeError(error, process.env.RUNPOD_MAIN_API_KEY)}\n`);
    process.exit(1);
  });
}

module.exports = {
  assertOutput,
  decodeCanonicalMp4,
  probeVideo,
  safeError,
  stripLarge,
  validateSpec,
};
