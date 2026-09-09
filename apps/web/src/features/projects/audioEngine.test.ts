import { describe, expect, it, vi } from "vitest";

import { withReferenceMatchParams } from "./audioEngine";
import { PreviewHost, type MovementSchedule } from "./audioEngine";
import { monitorMastering } from "./masterPreview";
import { bypassMatchDb, correctionGain } from "./audioAnalysis";
import { resolveDeliveryTarget } from "./masteringProfiles";
import { TEST_ENGINE_CONSTANTS } from "./engineConstants.fixture";
import { createPreviewProgramme } from "./previewProgramme";
import type { ProjectStem } from "@/api";
import { movementAt } from "./wasmEngine/movementSchedule";

const TARGETS = TEST_ENGINE_CONSTANTS.deliveryTargets;
const FALLBACK = TEST_ENGINE_CONSTANTS.deliveryDefault;

describe("resolveDeliveryTarget", () => {
  it("uses the served default with no preset named and no override set", () => {
    expect(resolveDeliveryTarget(undefined, TARGETS, FALLBACK)).toEqual(FALLBACK);
    expect(resolveDeliveryTarget({ target_preset: null }, TARGETS, FALLBACK)).toEqual(FALLBACK);
  });

  it("takes both numbers and the tolerance from a named target", () => {
    expect(resolveDeliveryTarget({ target_preset: "ebu-r128" }, TARGETS, FALLBACK)).toEqual({
      target_lkfs: -23,
      max_tp_dbtp: -1,
      tolerance_lu: 0.5,
    });
  });

  it("lets an explicit field override the preset one at a time", () => {
    const resolved = resolveDeliveryTarget(
      { target_preset: "ebu-r128", target: -20 },
      TARGETS,
      FALLBACK,
    );
    expect(resolved.target_lkfs).toBe(-20);
    expect(resolved.max_tp_dbtp).toBe(-1);
    expect(resolved.tolerance_lu).toBe(0.5);
  });

  it("falls back rather than guessing when the preset name is unknown", () => {
    expect(resolveDeliveryTarget({ target_preset: "atmos" }, TARGETS, FALLBACK)).toEqual(
      FALLBACK,
    );
  });
});

describe("loudness-matched bypass", () => {
  const DELIVERY = { target_lkfs: -18, max_tp_dbtp: -1 };
  const MAX_GAIN_DB = 30;
  // Normalizes cleanly to the target: +2 dB of gain still leaves 2 dB of
  // true-peak headroom.
  const MASTERED = { lkfs: -20, dbtp: -3 };
  // The unmastered side has no limiter, so its ceiling clamp bites first and
  // leaves it 5 LU quiet even after normalization.
  const BYPASSED = { lkfs: -24, dbtp: -2 };

  it("stays at unity until both sides are measured", () => {
    expect(bypassMatchDb(MASTERED, undefined, DELIVERY, MAX_GAIN_DB, true)).toBe(0);
    expect(bypassMatchDb(undefined, BYPASSED, DELIVERY, MAX_GAIN_DB, true)).toBe(0);
  });

  it("closes the gap the true-peak ceiling leaves between the two sides", () => {
    const mastered = -20 + 20 * Math.log10(correctionGain(MASTERED, DELIVERY, MAX_GAIN_DB, true));
    const bypassed = -24 + 20 * Math.log10(correctionGain(BYPASSED, DELIVERY, MAX_GAIN_DB, true));
    expect(mastered).toBeCloseTo(-18, 6);
    expect(bypassed).toBeCloseTo(-23, 6);
    expect(bypassMatchDb(MASTERED, BYPASSED, DELIVERY, MAX_GAIN_DB, true)).toBeCloseTo(5, 6);
  });

  it("matches the raw measurements when normalization is off", () => {
    expect(bypassMatchDb(MASTERED, BYPASSED, DELIVERY, MAX_GAIN_DB, false)).toBeCloseTo(4, 6);
  });

  it("is a no-op when both sides land on the target", () => {
    const same = { lkfs: -20, dbtp: -6 };
    expect(bypassMatchDb(same, same, DELIVERY, MAX_GAIN_DB, true)).toBeCloseTo(0, 12);
  });
});

describe("withReferenceMatchParams", () => {
  it("appends strength/max_db as the first query params", () => {
    expect(withReferenceMatchParams("/api/v1/projects/1/reference-match/fir", 0.5, 4)).toBe(
      "/api/v1/projects/1/reference-match/fir?strength=0.5&max_db=4",
    );
  });

  it("appends with & when the base url already carries a query param", () => {
    expect(withReferenceMatchParams("/fir?v=2", 1, 6)).toBe("/fir?v=2&strength=1&max_db=6");
  });

  it("leaves unset realization controls off the url", () => {
    expect(withReferenceMatchParams("/fir", 1, 6, null, null, null)).toBe(
      "/fir?strength=1&max_db=6",
    );
  });

  it("appends only the realization controls that are set", () => {
    expect(withReferenceMatchParams("/fir", 1, 6, null, 300, null)).toBe(
      "/fir?strength=1&max_db=6&low_hz=300",
    );
    expect(withReferenceMatchParams("/fir", 1, 6, 0.5, 300, 9000)).toBe(
      "/fir?strength=1&max_db=6&smooth_oct=0.5&low_hz=300&high_hz=9000",
    );
  });
});

describe("programme updates during movement compilation", () => {
  it("retries a position schedule invalidated by a later ordinary mix edit", async () => {
    const host = new PreviewHost({
      onReady: () => {}, onLoadProgress: () => {}, onError: () => {}, onPlaying: () => {},
      onCurrentTime: () => {}, onDuration: () => {}, onMeasuring: () => {},
      onMeasureProgress: () => {}, onLoudness: () => {}, onMaxChannels: () => {},
      onVolume: () => {}, onMuted: () => {}, onLoop: () => {}, onEngineStatus: () => {},
    });
    host.setConstants(TEST_ENGINE_CONSTANTS);
    const programme = (azimuth_deg: number, ceiling = 0) => createPreviewProgramme({
      stems: [{ id: "guitar", stem_key: "Guitar", audio_url: "/guitar.wav", channels: 2 } as ProjectStem],
      mix: { stem_placement: { Guitar: { azimuth_deg, elevation_deg: 0, width_deg: 0, object_size: 0 } } },
      mastering: { loudness: { max_tp: ceiling } },
      layoutChannels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      movementFeaturesUrl: "/movement-features",
    });
    host.setProgramme(programme(0));
    let finish!: (schedule: MovementSchedule) => void;
    const compile = vi.fn(() => new Promise<MovementSchedule>((resolve) => { finish = resolve; }));
    const schedule = (revision: number, x: number): MovementSchedule => ({
      version: 1, revision, sample_rate: 48000, duration_frames: 48000,
      grid_us: 20000, interpolation_us: 5208,
      stems: [{ stem_key: "Guitar", stem_index: 0, events: [{
        time_us: 0, position: [x, 1, 0], gains: x ? [1, 0] : [0, 1], interpolation_us: 5208,
      }] }],
    });
    const updates: MovementSchedule[] = [];
    Object.assign(host, {
      duration: 1, movementSchedule: schedule(1, 0),
      movementWasmModule: Promise.resolve({}), movementCompiler: { compile },
      nativeClient: { updateParams: (_params: unknown, _assets: unknown, _renderer: unknown, _tracking: unknown, next: MovementSchedule) => {
        updates.push(next);
        (host as unknown as { commitMovementSchedule(revision: number): void }).commitMovementSchedule(next.revision);
      } },
    });
    host.setProgramme(programme(75));
    expect(updates).toHaveLength(0);
    const pending = host.syncProgram();
    await Promise.resolve();
    host.setProgramme(programme(75, -1));
    finish(schedule(3, -1));
    await pending;
    expect(host.installedMovementSchedule?.revision).toBe(1);
    const retry = host.syncProgram();
    await Promise.resolve();
    expect(compile).toHaveBeenCalledTimes(2);
    finish(schedule(5, -1));
    await retry;
    expect(updates.at(-1)?.stems[0].events[0].gains).toEqual([1, 0]);
    expect(movementAt(host.installedMovementSchedule, "Guitar", 0)?.position).toEqual([-1, 1, 0]);
  });
  it("keeps movement while panner parameters update", () => {
    const updates: { params: Record<string, unknown>; schedule: MovementSchedule | null; ready: boolean }[] = [];
    const host = new PreviewHost({
      onReady: () => {}, onLoadProgress: () => {}, onError: () => {}, onPlaying: () => {},
      onCurrentTime: () => {}, onDuration: () => {}, onMeasuring: () => {},
      onMeasureProgress: () => {}, onLoudness: () => {}, onMaxChannels: () => {},
      onVolume: () => {}, onMuted: () => {}, onLoop: () => {}, onEngineStatus: () => {},
    });
    host.setConstants(TEST_ENGINE_CONSTANTS);
    host.setProgramme(createPreviewProgramme({
      stems: [{ id: "guitar", stem_key: "Guitar", audio_url: "/guitar.wav", channels: 2 } as ProjectStem],
      mix: { stem_placement: { Guitar: { azimuth_deg: 0, elevation_deg: 0, width_deg: 0, object_size: 0 } } },
      layoutChannels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      movementFeaturesUrl: "/movement-features",
    }));
    Object.assign(host as object, { duration: 1 });
    const prepared = (host as unknown as { movementRequestForCurrentProgramme: () => { key: string } }).movementRequestForCurrentProgramme();
    const schedule = { revision: 1 } as MovementSchedule;
    Object.assign(host as object, {
      movementRequestKey: prepared.key,
      movementSchedule: schedule,
      movementReady: true,
      client: { updateParams: (params: Record<string, unknown>, schedule: MovementSchedule | null, ready: boolean) => updates.push({ params, schedule, ready }) },
    });

    host.setProgramme(createPreviewProgramme({
      stems: [{ id: "guitar", stem_key: "Guitar", audio_url: "/guitar.wav", channels: 2 } as ProjectStem],
      mix: { stem_placement: { Guitar: { azimuth_deg: 75, elevation_deg: 0, width_deg: 0, object_size: 0 } } },
      layoutChannels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      movementFeaturesUrl: "/movement-features",
    }));

    expect(updates).toHaveLength(1);
    expect(updates[0].schedule).toBe(schedule);
    expect(updates[0].ready).toBe(true);
    expect((updates[0].params.stems as { object_placement: { azimuth_deg: number } }[])[0].object_placement.azimuth_deg).toBe(75);
  });

  it("does not prepare movement synchronously for a panner update", () => {
    const host = new PreviewHost({
      onReady: () => {}, onLoadProgress: () => {}, onError: () => {}, onPlaying: () => {},
      onCurrentTime: () => {}, onDuration: () => {}, onMeasuring: () => {},
      onMeasureProgress: () => {}, onLoudness: () => {}, onMaxChannels: () => {},
      onVolume: () => {}, onMuted: () => {}, onLoop: () => {}, onEngineStatus: () => {},
    });
    host.setConstants(TEST_ENGINE_CONSTANTS);
    const programme = (azimuth_deg: number) => createPreviewProgramme({
      stems: [{ id: "guitar", stem_key: "Guitar", audio_url: "/guitar.wav", channels: 2 } as ProjectStem],
      mix: { stem_placement: { Guitar: { azimuth_deg, elevation_deg: 0, width_deg: 0, object_size: 0 } } },
      layoutChannels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      movementFeaturesUrl: "/movement-features",
    });
    host.setProgramme(programme(0));
    Object.assign(host as object, {
      movementRequestForCurrentProgramme: () => { throw new Error("synchronous movement preparation"); },
    });

    expect(() => host.setProgramme(programme(75))).not.toThrow();
  });

  it("keeps the schedule when solo or movement settings change", () => {
    const updates: { schedule: MovementSchedule | null; ready: boolean }[] = [];
    const host = new PreviewHost({
      onReady: () => {}, onLoadProgress: () => {}, onError: () => {}, onPlaying: () => {},
      onCurrentTime: () => {}, onDuration: () => {}, onMeasuring: () => {},
      onMeasureProgress: () => {}, onLoudness: () => {}, onMaxChannels: () => {},
      onVolume: () => {}, onMuted: () => {}, onLoop: () => {}, onEngineStatus: () => {},
    });
    host.setConstants(TEST_ENGINE_CONSTANTS);
    const programme = (mix: Record<string, unknown> = {}) => createPreviewProgramme({
      stems: [{ id: "guitar", stem_key: "Guitar", audio_url: "/guitar.wav", channels: 2 } as ProjectStem],
      mix,
      layoutChannels: ["FL", "FR", "C", "LFE", "SL", "SR"],
      movementFeaturesUrl: "/movement-features",
    });
    host.setProgramme(programme());
    Object.assign(host as object, { duration: 1 });
    const prepared = (host as unknown as { movementRequestForCurrentProgramme: () => { key: string } }).movementRequestForCurrentProgramme();
    const schedule = { revision: 1 } as MovementSchedule;
    Object.assign(host as object, {
      movementRequestKey: prepared.key,
      movementSchedule: schedule,
      movementReady: true,
      client: { updateParams: (_params: Record<string, unknown>, next: MovementSchedule | null, ready: boolean) => updates.push({ schedule: next, ready }) },
    });

    host.setProgramme(programme({
      stem_solo: ["Guitar"],
      stem_movement: { Guitar: { depth: 0.5 } },
    }));

    expect(updates).toEqual([{ schedule, ready: true }]);
  });
});

describe("monitorMastering", () => {
  const mastering = {
    loudness: { normalize: true },
    eq: { profile: "spatial-air" },
    match_reference: { fir_url: "/fir", spectrum: true, rms: true, rms_gain_db: 2 },
  };

  it("passes the block through when nothing is bypassed", () => {
    expect(monitorMastering(mastering, false)).toBe(mastering);
  });

  it("strips everything but loudness for the whole-chain bypass", () => {
    expect(monitorMastering(mastering, true)).toEqual({ loudness: mastering.loudness });
  });

  it("strips both reference-match stages for the stage-scoped bypass", () => {
    const out = monitorMastering(mastering, false, true);
    expect(out?.eq).toEqual(mastering.eq);
    expect(out?.match_reference).toEqual({
      fir_url: "/fir", spectrum: false, rms: false, rms_gain_db: 2,
    });
  });

  it("lets the whole-chain bypass win over the stage-scoped one", () => {
    expect(monitorMastering(mastering, true, true)).toEqual({ loudness: mastering.loudness });
  });
});
