import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { BedPannerWindow } from "./BedPannerWindow";
import type { StemPlacement } from "./wasmEngine/panner";

const PLACEMENT: StemPlacement = {
  azimuth_deg: 0,
  elevation_deg: 0,
  width_deg: 90,
  object_size: 0,
  diversity: 0,
  center_level_db: 0,
};
const CHANNELS = ["FL", "FR", "C", "LFE", "SL", "SR", "TFL", "TFR"];

describe("BedPannerWindow", () => {
  it("opens from the strip panner and edits Logic surround controls", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    const onRoute = vi.fn();
    const onAmbient = vi.fn();
    render(<BedPannerWindow stemName="Bass" placement={PLACEMENT} route={{ LFE: 1 }} channels={CHANNELS}
      inputChannels={2} maxElevationDeg={30} ambientRear={0.5} ambientHeight={0.5} onPlacement={onPlacement} onRoute={onRoute} onAmbient={onAmbient} />);

    await user.click(screen.getByRole("button", { name: "Bed panner" }));
    expect(screen.getByRole("dialog", { name: "Bed panner" })).toHaveAttribute("aria-modal", "false");
    expect(screen.getByText("Bass Surround Panner")).toBeInTheDocument();
    expect(document.querySelectorAll("[data-bed-channel]")).toHaveLength(2);

    fireEvent.keyDown(screen.getByRole("slider", { name: "Diversity" }), { key: "ArrowRight" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ diversity: 0.01 }));

    fireEvent.keyDown(screen.getByRole("slider", { name: "Center level" }), { key: "ArrowLeft" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ center_level_db: -0.5 }));

    fireEvent.keyDown(screen.getByRole("slider", { name: "LFE level" }), { key: "ArrowLeft" });
    expect(onRoute).toHaveBeenLastCalledWith({ LFE: expect.any(Number) });

    fireEvent.keyDown(screen.getByRole("slider", { name: "Ambience to height" }), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ height: 0.51 });

    fireEvent.keyDown(screen.getByRole("slider", { name: "Height cutoff" }), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ heightCutoffHz: expect.any(Number) });
  });

  it("uses the spherical puck radius for elevation and hides stereo spread on mono beds", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<BedPannerWindow stemName="Bass" placement={PLACEMENT} route={{}} channels={CHANNELS}
      inputChannels={1} maxElevationDeg={30} onPlacement={onPlacement} onRoute={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Bed panner" }));
    await user.click(screen.getByRole("button", { name: "Spherical" }));
    expect(document.querySelectorAll("[data-bed-radar-ring]")).toHaveLength(3);
    fireEvent.keyDown(screen.getByRole("group", { name: "Surround position" }), { key: "ArrowUp" });

    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ elevation_deg: 1 }));
    expect(screen.queryByRole("slider", { name: "Spread" })).not.toBeInTheDocument();
  });

  it("uses the height cutoff", async () => {
    const user = userEvent.setup();
    const onAmbient = vi.fn();
    render(<BedPannerWindow stemName="Bass" placement={PLACEMENT} route={{}} channels={CHANNELS}
      inputChannels={1} maxElevationDeg={30} ambientHeightCutoffHz={500}
      ambientHeightCrossoverHz={4000} onPlacement={vi.fn()} onRoute={vi.fn()} onAmbient={onAmbient} />);

    await user.click(screen.getByRole("button", { name: "Bed panner" }));
    expect(screen.getByRole("slider", { name: "Height cutoff" })).toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole("slider", { name: "Height cutoff" }), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ heightCutoffHz: expect.any(Number) });
  });

  it("edits revision-2 wet trim and height texture", async () => {
    const user = userEvent.setup();
    const onAmbient = vi.fn();
    render(<BedPannerWindow stemName="Bass" placement={PLACEMENT} route={{}} channels={CHANNELS}
      inputChannels={1} maxElevationDeg={30} ambientTrimDb={3} heightTexture={0.1}
      onPlacement={vi.fn()} onRoute={vi.fn()} onAmbient={onAmbient} />);

    await user.click(screen.getByRole("button", { name: "Bed panner" }));
    fireEvent.keyDown(screen.getByRole("slider", { name: "Ambience trim" }), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ ambientTrimDb: 3.5 });
    fireEvent.keyDown(screen.getByRole("slider", { name: "Height texture" }), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ heightTexture: 0.11 });
  });
});
