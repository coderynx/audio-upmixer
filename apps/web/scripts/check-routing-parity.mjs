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
if (failed) process.exit(1);
