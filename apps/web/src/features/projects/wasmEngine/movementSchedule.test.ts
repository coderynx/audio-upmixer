import { describe, expect, it } from "vitest";
import type { MovementSchedule } from "./engineTypes";
import { movementAt, scenePositionFromMovement } from "./movementSchedule";

const schedule: MovementSchedule = {
  version: 1,
  revision: 2,
  sample_rate: 44100,
  duration_frames: 44100,
  grid_us: 20000,
  interpolation_us: 5208,
  stems: [{
    stem_key: "Guitar@front",
    stem_index: 0,
    events: [
      {
        time_us: 0,
        position: [0, 0, 0] as [number, number, number],
        gains: [1, 0],
        right_position: [0, 0, 0] as [number, number, number],
        right_gains: [1, 0],
        interpolation_us: 5208,
      },
      {
        time_us: 20000,
        position: [1, 0, 0] as [number, number, number],
        gains: [0, 1],
        right_position: [0.5, 0, 0] as [number, number, number],
        right_gains: [0.5, 0.5],
        interpolation_us: 5208,
      },
    ],
  }],
};

describe("movement schedule reader", () => {
  it("uses exact zone identity and shared gain transition timing", () => {
    const at = movementAt(schedule, "Guitar@front", 0.020 + 0.002604);
    expect(at?.position[0]).toBeCloseTo(0.5, 6);
    expect(at?.gains).toEqual([0.5, 0.5]);
    expect(movementAt(schedule, "Guitar@rear", 0.02)).toBeUndefined();
  });

  it("maps the Rust Cartesian convention to the scene axes", () => {
    expect(scenePositionFromMovement([0.25, -0.5, 0.75])).toEqual({
      x: 0.25,
      y: 0.75,
      z: 0.5,
    });
  });
});
