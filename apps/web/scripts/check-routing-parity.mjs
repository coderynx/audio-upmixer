#!/usr/bin/env node
/*
 * Compare the shipped WASM stream route with StemRouter on one f32 fixture.
 *
 * This is routing-only: Python supplies route_scale, WASM uses master:{}, and
 * no independent normalization or mastering parity is measured here.
 */
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const webRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const repoRoot = resolve(webRoot, "../..");
const SAMPLE_RATE = 48_000;
const FRAMES = Number.parseInt(process.env.PARITY_FRAMES ?? "4096", 10);
const BLOCK = 128;
const CHANNELS = ["FL", "FR", "C", "LFE", "BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR"];
const SHAPES = [
  "left", "right", "mono", "mono", "surround_left", "surround_right",
  "surround_left", "surround_right", "height_left", "height_right",
  "height_left", "height_right",
];
const GROUP_GAIN = (channel) => {
  if (channel === "C") return 0.85;
  if (["BL", "BR"].includes(channel)) return 0.55;
  if (["SL", "SR"].includes(channel)) return 0.6;
  if (["TFL", "TFR", "TBL", "TBR"].includes(channel)) return 0.55;
  return 1;
};

if (!Number.isInteger(FRAMES) || FRAMES < 1) {
  throw new Error("PARITY_FRAMES must be a positive integer");
}

function source(mono = false) {
  const left = new Float32Array(FRAMES);
  const right = new Float32Array(FRAMES);
  for (let i = 0; i < FRAMES; i += 1) {
    left[i] = ((i * 37 + 13) % 1001 - 500) / 1000;
    right[i] = mono ? left[i] : ((i * 97 + 271) % 1001 - 500) / 1000;
  }
  return { left, right };
}

function params(revision, rear, height, routeScale, trimDb = 0, texture = 0, cutoffHz = 1400) {
  return {
    speakers: CHANNELS.map((name, i) => ({
      name,
      azimuth_rad: i * 0.1,
      elevation_rad: 0,
      group_gain: GROUP_GAIN(name),
    })),
    lfe_index: 3,
    shapes: SHAPES,
    surround_downmix_coeff: 0.7071,
    height_downmix_coeff: 0.7071,
    sends: {
      surround_bass_cutoff_hz: 250,
      height_low_rolloff_hz: 150,
      height_low_rolloff_gain: 0.15,
      height_crossover_hz: 3000,
      height_high_shelf_gain: 1.5,
      height_directional_band_hz: 8000,
      height_directional_band_gain: 1,
      lfe_cutoff_hz: 120,
      lfe_filter_order: 4,
      lfe_gain: 0.31622776601683794,
    },
    stems: [{
      routing: [["FL", 1], ["FR", 1]],
      route_scale: routeScale,
      ambient_rear: rear,
      ambient_height: height,
      ambient_trim_db: trimDb,
      height_texture: texture,
      ambient_height_crossover_hz: 2100,
      ambient_height_cutoff_hz: cutoffHz,
      enabled: true,
    }],
    master: {},
    output_mode: "native",
  };
}

function instantiate() {
  const wasmPath = process.env.PARITY_WASM_PATH
    ? resolve(repoRoot, process.env.PARITY_WASM_PATH)
    : resolve(webRoot, "public/wasm/upmixer_dsp.wasm");
  const bytes = readFileSync(wasmPath);
  return new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
}

function writeF32(wasm, values) {
  const bytes = values.length * 4;
  const ptr = wasm.dsp_alloc(bytes);
  new Float32Array(wasm.memory.buffer, ptr, values.length).set(values);
  return { ptr, bytes };
}

function render(wasm, config, stem) {
  const encoded = new TextEncoder().encode(JSON.stringify(config));
  const configPtr = wasm.dsp_alloc(encoded.length);
  new Uint8Array(wasm.memory.buffer, configPtr, encoded.length).set(encoded);
  const engine = wasm.dsp_engine_new(SAMPLE_RATE, configPtr, encoded.length);
  wasm.dsp_free(configPtr, encoded.length);
  if (!engine) throw new Error("WASM rejected parity parameters");
  const left = writeF32(wasm, stem.left);
  const right = writeF32(wasm, stem.right);
  wasm.dsp_engine_add_stem(engine, left.ptr, right.ptr, FRAMES);
  wasm.dsp_free(left.ptr, left.bytes);
  wasm.dsp_free(right.ptr, right.bytes);

  const outPtr = wasm.dsp_alloc(CHANNELS.length * BLOCK * 4);
  const channels = CHANNELS.map(() => []);
  for (;;) {
    const written = wasm.dsp_engine_render(engine, outPtr, CHANNELS.length, BLOCK);
    if (!written) break;
    const block = new Float32Array(wasm.memory.buffer, outPtr, CHANNELS.length * BLOCK);
    for (let channel = 0; channel < CHANNELS.length; channel += 1) {
      for (let frame = 0; frame < written; frame += 1) {
        channels[channel].push(block[channel * BLOCK + frame]);
      }
    }
  }
  wasm.dsp_free(outPtr, CHANNELS.length * BLOCK * 4);
  wasm.dsp_engine_free(engine);
  return channels.map((channel) => Float32Array.from(channel));
}

function installMovementSchedule(wasm, engine, schedule, paramsPtr, paramsLength) {
  const encoded = new TextEncoder().encode(JSON.stringify(schedule));
  const schedulePtr = wasm.dsp_alloc(encoded.length);
  new Uint8Array(wasm.memory.buffer, schedulePtr, encoded.length).set(encoded);
  try {
    if (!wasm.dsp_engine_set_params_and_movement(engine, paramsPtr, paramsLength, schedulePtr, encoded.length)) {
      throw new Error("WASM movement schedule installation failed");
    }
  } finally {
    wasm.dsp_free(schedulePtr, encoded.length);
  }
}

function renderMovement(wasm, config, schedule, audio, keys, channels, blockSize) {
  // The schedule is deliberately absent from the JSON params. It crosses the
  // same sibling buffer setter used by the worklet so this parity check also
  // exercises the live transport boundary.
  const { movement_schedule: _movementSchedule, ...engineConfig } = config;
  const encoded = new TextEncoder().encode(JSON.stringify(engineConfig));
  const configPtr = wasm.dsp_alloc(encoded.length);
  new Uint8Array(wasm.memory.buffer, configPtr, encoded.length).set(encoded);
  const engine = wasm.dsp_engine_new(SAMPLE_RATE, configPtr, encoded.length);
  if (!engine) throw new Error("WASM rejected movement parity parameters");
  installMovementSchedule(wasm, engine, schedule, configPtr, encoded.length);
  wasm.dsp_free(configPtr, encoded.length);
  for (const key of keys) {
    const stem = audio[key];
    const left = Float32Array.from(stem.left);
    const right = Float32Array.from(stem.right);
    const leftMem = writeF32(wasm, left);
    const rightMem = writeF32(wasm, right);
    wasm.dsp_engine_add_stem(engine, leftMem.ptr, rightMem.ptr, left.length);
    wasm.dsp_free(leftMem.ptr, leftMem.bytes);
    wasm.dsp_free(rightMem.ptr, rightMem.bytes);
  }
  const outBytes = channels.length * blockSize * 4;
  const outPtr = wasm.dsp_alloc(outBytes);
  const output = channels.map(() => []);
  for (;;) {
    const written = wasm.dsp_engine_render(engine, outPtr, channels.length, blockSize);
    if (!written) break;
    const block = new Float32Array(wasm.memory.buffer, outPtr, channels.length * blockSize);
    for (let channel = 0; channel < channels.length; channel += 1) {
      for (let frame = 0; frame < written; frame += 1) {
        output[channel].push(block[channel * blockSize + frame]);
      }
    }
  }
  wasm.dsp_free(outPtr, outBytes);
  wasm.dsp_engine_free(engine);
  return output.map((channel) => Float32Array.from(channel));
}

function compileMovement(wasm, request) {
  const encoded = new TextEncoder().encode(JSON.stringify(request));
  const requestPtr = wasm.dsp_alloc(encoded.length);
  new Uint8Array(wasm.memory.buffer, requestPtr, encoded.length).set(encoded);
  try {
    const handle = wasm.dsp_movement_compile_json(requestPtr, encoded.length);
    if (!handle) throw new Error("WASM movement compiler returned no result");
    try {
      const resultPtr = wasm.dsp_movement_result_ptr(handle);
      const resultLength = wasm.dsp_movement_result_len(handle);
      if (!resultPtr || !resultLength) throw new Error("WASM movement compiler returned an empty result");
      const result = JSON.parse(new TextDecoder().decode(
        new Uint8Array(wasm.memory.buffer, resultPtr, resultLength),
      ));
      if (typeof result.error === "string") throw new Error(result.error);
      return result;
    } finally {
      wasm.dsp_movement_result_free(handle);
    }
  } finally {
    wasm.dsp_free(requestPtr, encoded.length);
  }
}

function movementDifference(expected, actual, path = "$") {
  if (expected === null || actual === null) {
    if (expected !== actual) throw new Error(`${path}: movement result shape mismatch`);
    return { maxAbs: 0, path };
  }
  if (typeof expected === "number" && typeof actual === "number") {
    return { maxAbs: Math.abs(expected - actual), path };
  }
  if (typeof expected !== typeof actual) {
    throw new Error(`${path}: movement result shape mismatch`);
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || expected.length !== actual.length) {
      throw new Error(`${path}: movement result length mismatch`);
    }
    return expected.reduce((best, value, index) => {
      const next = movementDifference(value, actual[index], `${path}[${index}]`);
      return next.maxAbs > best.maxAbs ? next : best;
    }, { maxAbs: 0, path });
  }
  if (typeof expected === "object") {
    const expectedKeys = Object.keys(expected).sort();
    const actualKeys = Object.keys(actual).sort();
    if (expectedKeys.join("\0") !== actualKeys.join("\0")) {
      throw new Error(`${path}: movement result fields mismatch`);
    }
    return expectedKeys.reduce((best, key) => {
      const next = movementDifference(expected[key], actual[key], `${path}.${key}`);
      return next.maxAbs > best.maxAbs ? next : best;
    }, { maxAbs: 0, path });
  }
  if (expected !== actual) throw new Error(`${path}: movement result value mismatch`);
  return { maxAbs: 0, path };
}

const fixture = JSON.parse(execFileSync("uv", ["run", "python", resolve(webRoot, "scripts/routing-parity-fixture.py")], {
  cwd: repoRoot,
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
}));
if (fixture.sample_rate !== SAMPLE_RATE || fixture.frames !== FRAMES) {
  throw new Error(`fixture shape mismatch: ${fixture.sample_rate} Hz, ${fixture.frames} frames`);
}
const wasm = instantiate();
const tolerance = 1e-6;
let failed = false;
for (const [name, expectedCase] of Object.entries(fixture.cases)) {
  const revision = Number(name.match(/^rev([12])-/)?.[1]);
  if (![1, 2].includes(revision)) throw new Error(`unrecognized parity case ${name}`);
  const rear = expectedCase.rear;
  const height = expectedCase.height;
  const trimDb = expectedCase.trim_db ?? 0;
  const texture = expectedCase.texture ?? 0;
  const cutoffHz = expectedCase.cutoff_hz ?? 1400;
  const actual = render(
    wasm,
    params(revision, rear, height, expectedCase.route_scale, trimDb, texture, cutoffHz),
    source(expectedCase.mono ?? false),
  );
  let maxAbs = 0;
  let sumSq = 0;
  let count = 0;
  const rearPeak = Math.max(...expectedCase.channels.slice(4, 8).flat().map(Math.abs));
  const heightPeak = Math.max(...expectedCase.channels.slice(8, 12).flat().map(Math.abs));
  const expectsRear = rear > 0;
  const expectsHeight = height > 0 || texture > 0;
  if ((expectsRear && rearPeak <= 1e-8) || (expectsHeight && heightPeak <= 1e-8)) {
    throw new Error(`${name}: requested ambient feed is silent (rear ${rearPeak}, height ${heightPeak})`);
  }
  for (let channel = 0; channel < CHANNELS.length; channel += 1) {
    const expected = Float32Array.from(expectedCase.channels[channel]);
    if (expected.length !== actual[channel].length) throw new Error(`${name}: frame count mismatch`);
    for (let frame = 0; frame < expected.length; frame += 1) {
      const delta = actual[channel][frame] - expected[frame];
      maxAbs = Math.max(maxAbs, Math.abs(delta));
      sumSq += delta * delta;
      count += 1;
    }
  }
  const rms = Math.sqrt(sumSq / count);
  const ok = maxAbs <= tolerance;
  failed ||= !ok;
  console.log(`${ok ? "ok" : "FAIL"} ${name.padEnd(14)} route_scale ${expectedCase.route_scale.toFixed(9)} rear_peak ${rearPeak.toExponential(3)} height_peak ${heightPeak.toExponential(3)} max_abs ${maxAbs.toExponential(3)} rms ${rms.toExponential(3)}`);
}

for (const [layout, expectedCase] of Object.entries(fixture.movement_cases?.layouts ?? {})) {
  const actual = compileMovement(wasm, expectedCase.request);
  const comparison = movementDifference(expectedCase.schedule, actual);
  if (comparison.maxAbs > tolerance) {
    throw new Error(`movement ${layout}: max_abs ${comparison.maxAbs} at ${comparison.path}`);
  }
  console.log(`ok movement-${layout.padEnd(6)} canonical schedule max_abs ${comparison.maxAbs.toExponential(3)}`);
  const channels = expectedCase.request.channels;
  for (const [caseName, movementCase] of Object.entries(expectedCase.cases ?? {})) {
    for (const blockSize of expectedCase.block_sizes ?? []) {
      const rendered = renderMovement(
        wasm,
        movementCase.params,
        expectedCase.schedule,
        fixture.movement_cases.audio,
        expectedCase.audio_keys,
        channels,
        blockSize,
      );
      let maxAbs = 0;
      let maxChannel = 0;
      let maxFrame = 0;
      let maxExpected = 0;
      let maxActual = 0;
      let sumSq = 0;
      let count = 0;
      for (let channel = 0; channel < channels.length; channel += 1) {
        const expected = Float32Array.from(movementCase.expected[channel]);
        if (expected.length !== rendered[channel].length) {
          throw new Error(`movement ${layout}/${caseName}/${blockSize}: frame count mismatch`);
        }
        for (let frame = 0; frame < expected.length; frame += 1) {
          const delta = rendered[channel][frame] - expected[frame];
          if (Math.abs(delta) > maxAbs) {
            maxAbs = Math.abs(delta);
            maxChannel = channel;
            maxFrame = frame;
            maxExpected = expected[frame];
            maxActual = rendered[channel][frame];
          }
          sumSq += delta * delta;
          count += 1;
        }
      }
      const rms = Math.sqrt(sumSq / Math.max(1, count));
      const ok = maxAbs <= tolerance;
      failed ||= !ok;
      console.log(
        `${ok ? "ok" : "FAIL"} movement-${layout.padEnd(6)} ${caseName.padEnd(12)} ` +
        `block ${String(blockSize).padStart(4)} ragged max_abs ${maxAbs.toExponential(3)} ` +
        `rms ${rms.toExponential(3)}` +
        (ok ? "" : ` at ${channels[maxChannel]}[${maxFrame}] expected ${maxExpected.toExponential(3)} actual ${maxActual.toExponential(3)}`),
      );
    }
  }
  for (const [lockName, collapseModes] of Object.entries(expectedCase.collapse_cases ?? {})) {
    for (const [mode, collapseCase] of Object.entries(collapseModes)) {
      for (const blockSize of [127, 511, 1024]) {
        const rendered = renderMovement(
          wasm,
          collapseCase.params,
          expectedCase.schedule,
          fixture.movement_cases.audio,
          expectedCase.audio_keys,
          ["FL", "FR"],
          blockSize,
        );
        let maxAbs = 0;
        let maxChannel = 0;
        let maxFrame = 0;
        let maxExpected = 0;
        let maxActual = 0;
        let sumSq = 0;
        let count = 0;
        for (let channel = 0; channel < 2; channel += 1) {
          const expected = Float32Array.from(collapseCase.expected[channel]);
          if (expected.length !== rendered[channel].length) {
            throw new Error(`movement ${layout}/${lockName}/${mode}/${blockSize}: frame count mismatch`);
          }
          for (let frame = 0; frame < expected.length; frame += 1) {
            const delta = rendered[channel][frame] - expected[frame];
            if (Math.abs(delta) > maxAbs) {
              maxAbs = Math.abs(delta);
              maxChannel = channel;
              maxFrame = frame;
              maxExpected = expected[frame];
              maxActual = rendered[channel][frame];
            }
            sumSq += delta * delta;
            count += 1;
          }
        }
        const rms = Math.sqrt(sumSq / Math.max(1, count));
        const ok = maxAbs <= tolerance;
        failed ||= !ok;
        console.log(
          `${ok ? "ok" : "FAIL"} movement-${layout.padEnd(6)} ${lockName.padEnd(12)} ` +
          `${mode.padEnd(9)} block ${String(blockSize).padStart(4)} collapse ` +
          `max_abs ${maxAbs.toExponential(3)} rms ${rms.toExponential(3)}` +
          (ok ? "" : ` at [${maxChannel}][${maxFrame}] expected ${maxExpected.toExponential(3)} actual ${maxActual.toExponential(3)}`),
        );
      }
    }
  }
}
if (failed) process.exit(1);
