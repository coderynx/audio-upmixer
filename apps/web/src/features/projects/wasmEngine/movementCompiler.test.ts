import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  MovementCompilerClient,
  type MovementCompileRequest,
} from "./movementCompiler";
import { compileMovement } from "./movementWorkerRuntime";

function request(revision: number): MovementCompileRequest {
  return {
    sample_rate: 48_000,
    duration_frames: 48_000,
    revision,
    channels: ["FL", "FR"],
    stems: [],
  };
}

describe("MovementCompilerClient", () => {
  it("merges deltas even when their base result was superseded", async () => {
    const worker = { postMessage: vi.fn(), terminate: vi.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: ErrorEvent) => void) | null };
    const client = new MovementCompilerClient(() => worker);
    const first = client.compile("/features", request(1)).catch(() => null);
    const latest = client.compile("/features", request(2));
    const guitar = { stem_key: "Guitar", stem_index: 0, events: [] };
    const vocals = { stem_key: "Vocals", stem_index: 1, events: [] };
    worker.onmessage?.({ data: { type: "compiled", id: 1,
      schedule: { revision: 1, stems: [guitar, vocals] } } } as MessageEvent);
    const moved = { ...guitar, events: [{ time_us: 0, position: [-1, 0, 0], gains: [1, 0], interpolation_us: 5208 }] };
    worker.onmessage?.({ data: { type: "compiled", id: 2, baseRevision: 1,
      schedule: { revision: 2, stems: [moved] } } } as MessageEvent);
    expect(await first).toBeNull();
    const result = await latest;
    expect(result.stems).toEqual([moved, vocals]);
    expect(result.stems[1]).toBe(vocals);
    client.dispose();
  });
  it("compiles only the latest queued position after an in-flight request", async () => {
    const worker = {
      postMessage: vi.fn(), terminate: vi.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: ErrorEvent) => void) | null,
    };
    const client = new MovementCompilerClient(() => worker);
    const results = Array.from({ length: 100 }, (_, index) =>
      client.compile("/features.json", request(index + 1)).catch(() => null));
    expect(worker.postMessage).toHaveBeenCalledTimes(1);
    const finish = (index: number) => {
      const message = worker.postMessage.mock.calls[index][0];
      worker.onmessage?.({ data: { type: "compiled", id: message.id,
        schedule: { revision: message.request.revision, stems: [] },
      } } as MessageEvent);
    };
    finish(0);
    expect(worker.postMessage).toHaveBeenCalledTimes(2);
    expect(worker.postMessage.mock.calls[1][0].request.revision).toBe(100);
    finish(1);
    expect((await Promise.all(results)).slice(0, -1)).toEqual(Array(99).fill(null));
    await expect(results[99]).resolves.toMatchObject({ revision: 100 });
    client.dispose();
  });

  it("runs the latest request even when the superseded worker job fails", async () => {
    const worker = {
      postMessage: vi.fn(),
      terminate: vi.fn(),
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: ErrorEvent) => void) | null,
    };
    const client = new MovementCompilerClient(() => worker);
    const first = client.compile("/features.json", request(1));
    const second = client.compile("/features.json", request(2));

    await expect(first).rejects.toThrow("stale movement compilation");
    worker.onmessage?.({ data: { type: "error", id: worker.postMessage.mock.calls[0][0].id,
      message: "old compilation failed",
    } } as MessageEvent);
    const message = worker.postMessage.mock.calls[1][0];
    worker.onmessage?.({
      data: {
        type: "compiled",
        id: message.id,
        schedule: {
          version: 1,
          revision: 2,
          sample_rate: 48_000,
          duration_frames: 48_000,
          grid_us: 20_000,
          interpolation_us: 5_208,
          stems: [],
        },
      },
    } as MessageEvent);
    await expect(second).resolves.toMatchObject({ revision: 2 });
    client.dispose();
  });

  it("validates a prepared superset and reuses it across proxy URL changes", async () => {
    const features = {
      version: 1,
      sample_rate: 44_100,
      frame_count: 88_200,
      window_frames: 441,
      energies: Array.from({ length: 200 }, () => 0.04),
    };
    const sidecar = {
      version: 1,
      sample_rate: 44_100,
      frame_count: 88_200,
      // The prepared sidecar may retain unselected instrument and parent stems.
      stems: [
        "Crowd", "Other", "Guitar", "Bass", "Kick", "Snare", "Toms",
        "Hi-Hat", "Ride", "Crash", "Lead Vocals", "Backing Vocals",
        "Piano", "Vocals", "Drums",
      ].map((stem_key) => ({ stem_key, features })),
    };
    const tuning = {
      activity_floor_db: -65, activity_enter_db: 6, activity_leave_db: 3,
      activity_dwell_ms: 80, activity_exit_dwell_ms: 250,
      focus_prominence_db: 6, focus_share: 0.85, focus_qualification_ms: 750,
      focus_attack_ms: 750, focus_release_ms: 1500, vocal_hold_ms: 1200,
      winner_hold_ms: 1000, challenger_db: 3, supporting_span_db: 24,
      percussion_rise_db: 6, percussion_gate_db: 6,
      percussion_retrigger_ms: 120, percussion_attack_ms: 40,
      percussion_return_ms: 300, toms_return_ms: 500, crash_return_ms: 1000,
    };
    const request: MovementCompileRequest = {
      sample_rate: 48_000,
      duration_frames: 96_000,
      revision: 3,
      channels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      stems: [{
        stem_key: "Guitar",
        stem_name: "Guitar",
        gain_db: 0,
        enabled: true,
        included: true,
        placement: {
          azimuth_deg: 42,
          elevation_deg: 0,
          width_deg: 0,
          object_size: 0,
          lfe: 0,
          diversity: 0,
          center_level_db: 0,
        },
        home_gains: [0.5, 0.5, 0.25, 0.05, 0.35, 0.35],
        home_right_gains: [],
        settings: {
          enabled: true,
          role: "featured",
          depth: 0.4,
          response: 1,
          sensitivity: 0.5,
          start_s: 0,
          end_s: null,
        },
        object_mode: null,
        channel_lock: false,
        zone_exclusion: [],
      }],
      tuning,
    };
    const fetchFeatures = vi.fn(async () => ({
      ok: true,
      json: async () => sidecar,
    }));
    vi.stubGlobal("fetch", fetchFeatures);
    const module = new WebAssembly.Module(readFileSync(
      resolve(process.cwd(), "public/wasm/upmixer_dsp.wasm"),
    ));
    const first = await compileMovement({
      featuresUrl: "/features.json",
      proxyUrl: "/preview-low.ogg",
      request,
      wasmModule: module,
    });
    const second = await compileMovement({
      featuresUrl: "/features.json",
      proxyUrl: "/preview-high.ogg",
      request,
      wasmModule: module,
    });
    expect(fetchFeatures).toHaveBeenCalledTimes(1);
    expect(first.sample_rate).toBe(44_100);
    expect(first.duration_frames).toBe(88_200);
    expect(first.stems[0]?.events.length).toBeGreaterThan(1);
    expect(second).toEqual(first);

    // Live panning must not wait for another full-track analysis.
    const longFeatures = { ...features, frame_count: 44100 * 420,
      energies: Array.from({ length: 42000 }, (_, i) => 0.04 * (1 + Math.sin(i / 100))) };
    const longSidecar = { ...sidecar, frame_count: longFeatures.frame_count,
      stems: sidecar.stems.map((stem) => ({ ...stem, features: longFeatures })) };
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => longSidecar })));
    const longRequest = { ...request, stems: sidecar.stems.slice(0, 12).map(({ stem_key }) =>
      ({ ...request.stems[0], stem_key, stem_name: stem_key })) };
    await compileMovement({ featuresUrl: "/long-features.json", request: longRequest, wasmModule: module });
    const started = performance.now();
    const moved = await compileMovement({ featuresUrl: "/long-features.json", wasmModule: module,
      request: { ...longRequest, revision: 4, stems: longRequest.stems.map((stem) =>
        stem.stem_key === "Guitar" ? { ...stem, placement: { ...stem.placement, azimuth_deg: -75 } } : stem) } });
    expect(moved.revision).toBe(4);
    expect(performance.now() - started).toBeLessThan(50);

    const expectSidecarError = async (
      featuresUrl: string,
      stems: typeof sidecar.stems,
      message: string,
    ) => {
      vi.stubGlobal("fetch", vi.fn(async () => ({
        ok: true,
        json: async () => ({ ...sidecar, stems }),
      })));
      await expect(compileMovement({ featuresUrl, request, wasmModule: module }))
        .rejects.toThrow(message);
    };
    await expectSidecarError(
      "/missing.json",
      sidecar.stems.filter(({ stem_key }) => stem_key !== "Guitar"),
      "missing a requested stem identity",
    );
    await expectSidecarError(
      "/duplicate.json",
      [...sidecar.stems, { stem_key: "Guitar", features }],
      "duplicate stem identities",
    );
    await expectSidecarError(
      "/empty.json",
      [...sidecar.stems, { stem_key: "", features }],
      "identities must be non-empty",
    );
    vi.unstubAllGlobals();
  });
});
