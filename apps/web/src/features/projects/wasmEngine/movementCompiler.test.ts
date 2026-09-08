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
  it("drops an older worker result when a newer revision supersedes it", async () => {
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
