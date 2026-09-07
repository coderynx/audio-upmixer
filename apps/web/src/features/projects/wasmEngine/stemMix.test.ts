import { describe, expect, it } from "vitest";

import type { ProjectStem } from "@/api";
import { TEST_ENGINE_CONSTANTS } from "../engineConstants.fixture";
import { resolveStemMixes } from "./stemMix";

describe("resolveStemMixes", () => {
  it("seeds authored objects at unity until exact route normalization lands", () => {
    const stems = resolveStemMixes({
      stems: [
        { id: "v", stem_key: "Vocals" } as ProjectStem,
        { id: "b", stem_key: "Bass" } as ProjectStem,
      ],
      scene: { stems: {} },
      mix: {
        bed_trim_db: 3,
        stem_rebalance: { Vocals: 1, Bass: 2 },
        stem_routing: { Vocals: { C: 1, LFE: 1 }, Bass: { C: 1, LFE: 1 } },
        stem_object_mode: { Vocals: "linked-stereo" },
        stem_placement: {
          Vocals: { azimuth_deg: 0, elevation_deg: 0, width_deg: 0, object_size: 0 },
        },
      },
      stemEqTaps: new Map(),
      constants: TEST_ENGINE_CONSTANTS,
    });

    expect(stems[0].routing).toEqual({ C: 1 });
    expect(stems[1].routing).toEqual({ C: 1, LFE: 1 });
    expect(stems[0].routeScale).toBe(1);
    expect(stems[0].rebalanceDb).toBe(1);
    expect(stems[1].routeScale).toBeCloseTo(
      1 / TEST_ENGINE_CONSTANTS.channelGains.center,
      12,
    );
    expect(stems[1].rebalanceDb).toBe(5);
  });

  it("keeps the revision-2 cutoff independent from the legacy crossover", () => {
    const stems = resolveStemMixes({
      stems: [{ id: "v", stem_key: "Vocals" } as ProjectStem],
      scene: { stems: {} },
      mix: {
        stem_ambient_height_cutoff_hz: { Vocals: 750 },
        stem_ambient_height_crossover_hz: { Vocals: 4000 },
      },
      stemEqTaps: new Map(),
      constants: TEST_ENGINE_CONSTANTS,
    });

    expect(stems[0].ambientHeightCutoffHz).toBe(750);
    expect(stems[0].ambientHeightCrossoverHz).toBe(4000);
  });

  it("clamps revision-2 wet trim and height texture", () => {
    const stems = resolveStemMixes({
      stems: [{ id: "v", stem_key: "Vocals" } as ProjectStem],
      scene: { stems: {} },
      mix: {
        stem_ambient_trim_db: { Vocals: 9 },
        stem_height_texture: { Vocals: -1 },
      },
      stemEqTaps: new Map(),
      constants: TEST_ENGINE_CONSTANTS,
    });

    expect(stems[0].ambientTrimDb).toBe(6);
    expect(stems[0].heightTexture).toBe(0);
  });
});
