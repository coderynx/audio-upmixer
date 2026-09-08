import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { NativePreviewClient } from "./nativePreviewClient";
import type { MovementSchedule } from "./wasmEngine/engineTypes";

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => true,
  Channel: class { onmessage = () => {}; },
  invoke: vi.fn(async () => 1),
}));
beforeEach(() => { vi.mocked(invoke).mockReset().mockResolvedValue(1); });
afterEach(() => { vi.unstubAllGlobals(); });

it("does not serialize the installed movement schedule on each parameter adjustment", async () => {
  let frame = () => {};
  vi.stubGlobal("requestAnimationFrame", (callback: () => void) => { frame = callback; return 1; });
  const schedule: MovementSchedule = {
    version: 1, revision: 1, sample_rate: 48000, duration_frames: 48000 * 60,
    grid_us: 20000, interpolation_us: 5208,
    stems: [{ stem_key: "Vocals", stem_index: 0, events: Array.from({ length: 3000 }, (_, index) => ({
      time_us: index * 20000, position: [0, 0, 1], gains: [1, 0], interpolation_us: 5208,
    })) }],
  };
  const client = await NativePreviewClient.create({
    sources: [], params: {}, movementSchedule: schedule, assets: {}, renderer: "direct",
    appleHeadTracking: false, onMaxChannels: () => {}, onLoadProgress: () => {},
  }, {});
  let bytes = 0;
  vi.mocked(invoke).mockImplementation(async (_command, args) => {
    bytes += JSON.stringify(args).length;
    return undefined;
  });
  for (let gain = 0; gain < 10; gain++) {
    client.updateParams({ gain }, {}, "direct", false, schedule);
    frame();
    await vi.waitFor(() => expect(invoke).toHaveBeenCalledTimes(gain + 2));
  }
  expect(bytes).toBeLessThan(5000);
  client.dispose();
});

it("retries failed schedule transfers and orders replacements and clearing before transport", async () => {
  vi.stubGlobal("requestAnimationFrame", () => 1);
  const onError = vi.fn();
  const client = await NativePreviewClient.create({
    sources: [], params: {}, assets: {}, renderer: "direct", appleHeadTracking: false,
    onMaxChannels: () => {}, onLoadProgress: () => {},
  }, { onError });
  const schedule: MovementSchedule = {
    version: 1, revision: 2, sample_rate: 48000, duration_frames: 48000,
    grid_us: 20000, interpolation_us: 5208, stems: [],
  };
  vi.mocked(invoke).mockRejectedValueOnce(new Error("IPC failed"));
  client.updateParams({}, {}, "direct", false, schedule);
  await client.seek(0);
  expect(onError).toHaveBeenCalledWith("IPC failed");
  client.updateParams({}, {}, "direct", false, schedule);
  await client.seek(0);
  expect(vi.mocked(invoke).mock.calls.at(-2)).toEqual([
    "native_preview_update", { request: expect.objectContaining({ movementSchedule: schedule, movementUnchanged: false }) },
  ]);
  client.updateParams({}, {}, "direct", false, schedule);
  await client.seek(0);
  expect(vi.mocked(invoke).mock.calls.at(-2)?.[1]).toMatchObject({ request: { movementUnchanged: true } });
  client.updateParams({}, {}, "direct", false, null);
  await client.seek(0);
  expect(vi.mocked(invoke).mock.calls.at(-2)).toEqual([
    "native_preview_update", { request: expect.objectContaining({ movementSchedule: null, movementUnchanged: false }) },
  ]);
  client.dispose();
});
