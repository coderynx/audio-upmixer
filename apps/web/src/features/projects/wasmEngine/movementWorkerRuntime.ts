import type {
  CanonicalFeatures,
  MovementCompileRequest,
  MovementFeatureSidecar,
} from "./movementCompiler";
import type { MovementSchedule } from "./engineTypes";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

type WasmExports = {
  memory: WebAssembly.Memory;
  dsp_alloc(bytes: number): number;
  dsp_free(ptr: number, bytes: number): void;
  dsp_movement_compile_json(ptr: number, length: number): number;
  dsp_movement_result_ptr(handle: number): number;
  dsp_movement_result_len(handle: number): number;
  dsp_movement_result_free(handle: number): void;
};

let wasmPromise: Promise<WasmExports> | null = null;
let sidecarUrl = "";
let sidecarPromise: Promise<MovementFeatureSidecar> | null = null;

function loadWasm(module: WebAssembly.Module | undefined): Promise<WasmExports> {
  if (!wasmPromise) {
    if (!module) {
      throw new Error("movement compiler requires the loaded DSP WASM module");
    }
    wasmPromise = Promise.resolve(
      new WebAssembly.Instance(module).exports as unknown as WasmExports,
    );
  }
  return wasmPromise;
}

function loadSidecar(url: string): Promise<MovementFeatureSidecar> {
  if (sidecarPromise && sidecarUrl === url) return sidecarPromise;
  sidecarUrl = url;
  sidecarPromise = fetch(url, { cache: "no-store" }).then(async (response) => {
    if (!response.ok) throw new Error(`Failed to fetch movement features: ${response.status}`);
    return await response.json() as MovementFeatureSidecar;
  });
  return sidecarPromise;
}

/** Canonical features are keyed independently of any lossy preview proxy. */
export function loadCanonicalFeatures(
  featuresUrl: string,
  _proxyUrl?: string,
): Promise<MovementFeatureSidecar> {
  return loadSidecar(featuresUrl);
}

function validateSidecar(sidecar: MovementFeatureSidecar, expectedKeys: string[]): void {
  if (
    !sidecar
    || !Number.isInteger(sidecar.version)
    || sidecar.version !== 1
    || !Number.isInteger(sidecar.sample_rate)
    || sidecar.sample_rate <= 0
    || !Array.isArray(sidecar.stems)
  ) {
    throw new Error("movement feature sidecar has an unsupported header");
  }
  if (!Number.isInteger(sidecar.frame_count) || sidecar.frame_count < 0) {
    throw new Error("movement feature sidecar has an invalid duration");
  }
  const expected = [...expectedKeys].sort();
  const actual = sidecar.stems.map((stem) => stem.stem_key).sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error("movement feature sidecar stem identities do not match the prepared stems");
  }
  for (const entry of sidecar.stems) {
    if (
      !entry
      || typeof entry.stem_key !== "string"
      || !entry.features
      || !Array.isArray(entry.features.energies)
    ) {
      throw new Error("movement feature sidecar has an invalid stem entry");
    }
    const features = entry.features;
    const windowFrames = features.window_frames;
    const expectedWindowFrames = Math.max(1, Math.round(sidecar.sample_rate * 0.01));
    const expectedWindows = windowFrames > 0
      ? Math.ceil(features.frame_count / windowFrames)
      : -1;
    if (
      features.version !== sidecar.version
      || features.sample_rate !== sidecar.sample_rate
      || !Number.isInteger(features.window_frames)
      || features.window_frames <= 0
      || !Number.isInteger(features.frame_count)
      || features.frame_count < 0
      || features.frame_count > sidecar.frame_count
      || windowFrames !== expectedWindowFrames
      || features.energies.length !== expectedWindows
      || features.energies.some((energy) => !Number.isFinite(energy) || energy < 0)
    ) {
      throw new Error(`movement features are invalid for stem '${entry.stem_key}'`);
    }
  }
}

function featuresFor(sidecar: MovementFeatureSidecar, stemKey: string): CanonicalFeatures {
  const entry = sidecar.stems.find((stem) => stem.stem_key === stemKey);
  if (!entry) throw new Error(`Movement features are missing stem '${stemKey}'`);
  return entry.features;
}

export function withCanonicalFeatures(
  request: MovementCompileRequest,
  sidecar: MovementFeatureSidecar,
): MovementCompileRequest {
  return {
    ...request,
    // The canonical source rate and full prepared duration are authoritative;
    // preview PCM is resampled to the fixed 48 kHz render context.
    sample_rate: sidecar.sample_rate,
    duration_frames: sidecar.frame_count,
    feature_sidecar: sidecar,
    stems: request.stems.map((stem) => ({
      ...stem,
      features: featuresFor(sidecar, stem.stem_key),
    })),
  };
}

function readBytes(wasm: WasmExports, ptr: number, length: number): Uint8Array {
  return new Uint8Array(wasm.memory.buffer, ptr, length).slice();
}

/** Call the control-rate WASM entry point. It is deliberately separate from
 * the render engine ABI: schedule compilation never runs in an audio
 * callback. The result handle owns the serialized bytes until it is freed. */
function compileWithWasm(wasm: WasmExports, request: MovementCompileRequest): MovementSchedule {
  const input = encoder.encode(JSON.stringify(request));
  const inputPtr = wasm.dsp_alloc(input.byteLength);
  new Uint8Array(wasm.memory.buffer, inputPtr, input.byteLength).set(input);
  try {
    const handle = wasm.dsp_movement_compile_json(inputPtr, input.byteLength);
    if (!handle) throw new Error("movement compilation failed");
    try {
      const outputPtr = wasm.dsp_movement_result_ptr(handle);
      const outputLength = wasm.dsp_movement_result_len(handle);
      if (!outputPtr || !outputLength) throw new Error("movement compilation failed");
      const result = JSON.parse(decoder.decode(readBytes(wasm, outputPtr, outputLength))) as
        MovementSchedule & { error?: string };
      if (typeof result.error === "string") throw new Error(result.error);
      return result;
    } finally {
      wasm.dsp_movement_result_free(handle);
    }
  } finally {
    wasm.dsp_free(inputPtr, input.byteLength);
  }
}

export async function compileMovement(message: {
  featuresUrl: string;
  proxyUrl?: string;
  request: MovementCompileRequest;
  wasmModule?: WebAssembly.Module;
}) {
  const sidecar = await loadCanonicalFeatures(message.featuresUrl, message.proxyUrl);
  validateSidecar(sidecar, message.request.stems.map((stem) => stem.stem_key));
  const request = withCanonicalFeatures(message.request, sidecar);
  return compileWithWasm(await loadWasm(message.wasmModule), request);
}

type WorkerScope = {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(value: unknown): void;
};
const worker = self as unknown as WorkerScope;
worker.onmessage = (event: MessageEvent<{ type: string; id: number; featuresUrl: string; proxyUrl?: string; request: MovementCompileRequest; wasmModule?: WebAssembly.Module }>) => {
  if (event.data.type !== "compile") return;
  void compileMovement(event.data).then(
    (schedule) => worker.postMessage({ type: "compiled", id: event.data.id, schedule }),
    (error: unknown) => worker.postMessage({
      type: "error",
      id: event.data.id,
      message: error instanceof Error ? error.message : String(error),
    }),
  );
};
