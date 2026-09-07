import { describe, expect, it } from "vitest";
import { defaultManifest, normalizeManifest } from "./manifest";

describe("normalizeManifest", () => {
  it("fills missing nested defaults without losing supplied values", () => {
    const manifest = normalizeManifest({
      engine: { mode: "stem" },
      mixing: { bed_trim_db: 2.5, stem_source_anchor_strength: 0.35 },
    });
    expect(manifest.engine.mode).toBe("stem");
    expect(manifest.engine.stems).toEqual(defaultManifest.engine.stems);
    expect(manifest.engine.stem_ensemble).toBe(false);
    expect((manifest.engine as Record<string, unknown>).stem_native_rate).toBeUndefined();
    expect(manifest.mixing.stem_source_anchor_strength).toBe(0.35);
    expect(manifest.mixing.bed_trim_db).toBe(2.5);
    expect(manifest.mixing.stem_routing).toEqual(defaultManifest.mixing.stem_routing);
  });

  it("keeps enhancement defaults off", () => {
    expect(defaultManifest.mixing.stem_ambient_trim_db).toEqual({});
    expect(defaultManifest.mixing.stem_height_texture).toEqual({});
  });
});
