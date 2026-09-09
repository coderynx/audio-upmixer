import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { getStemColor } from "@/lib/stems";
import {
  ObjectPannerWindow,
  objectChannelCoordinates,
  objectChannelPositions,
  pannerCoordinatesFromPlacement,
  pannerToScenePosition,
  placementFromPannerCoordinates,
  placementFromPannerPosition,
} from "./ObjectPannerWindow";
import type { StemPlacement } from "./wasmEngine/panner";

const PLACEMENT: StemPlacement = { azimuth_deg: 0, elevation_deg: 0, width_deg: 0, object_size: 0.5 };

describe("object panner geometry", () => {
  it("derives distinct front channels from a front-centre 90 degree image", () => {
    const placement = { ...PLACEMENT, left_right: 0, back_front: 1, width_deg: 90 };

    const channels = objectChannelCoordinates(placement);
    expect(channels.left.leftRight).toBeCloseTo(-Math.SQRT1_2, 9);
    expect(channels.right.leftRight).toBeCloseTo(Math.SQRT1_2, 9);
    expect(channels.left.backFront).toBeCloseTo(Math.SQRT1_2, 9);
    expect(channels.right.backFront).toBeCloseTo(Math.SQRT1_2, 9);
  });

  it("keeps both channel identities at one coordinate when spread is zero", () => {
    const placement = { ...PLACEMENT, left_right: -0.25, back_front: 0.75 };
    const channels = objectChannelCoordinates(placement);

    expect(channels.left).toEqual(channels.right);
    expect(channels.left).toMatchObject({ leftRight: -0.25, backFront: 0.75 });
  });

  it("preserves signed coordinates and legacy azimuth compatibility", () => {
    const next = placementFromPannerCoordinates(PLACEMENT, { leftRight: 0.5, backFront: -0.25, elevation: 0.2 });

    expect(next).toMatchObject({ left_right: 0.5, back_front: -0.25 });
    expect(next.elevation_deg).toBeCloseTo(11.537, 3);
    expect(pannerCoordinatesFromPlacement(next)).toEqual({ leftRight: 0.5, backFront: -0.25, elevation: 0.2 });
    expect(next.azimuth_deg).toBeCloseTo(-116.565, 3);
  });

  it("maps canonical panner axes into scene axes once", () => {
    expect(pannerToScenePosition({ leftRight: -0.5, backFront: 1, elevation: 0.25 })).toEqual({
      x: -0.5, y: 0.25, z: -1,
    });
  });

  it("keeps channel labels attached after signed spread crosses", () => {
    const channels = objectChannelCoordinates({ ...PLACEMENT, left_right: 0, back_front: 1, width_deg: -90 });

    expect(channels.left.leftRight).toBeGreaterThan(0);
    expect(channels.right.leftRight).toBeLessThan(0);
  });
});

describe("ObjectPannerWindow", () => {
  it("moves horizontal and vertical placement from the panner controls", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={onPlacement} />);

    await user.click(screen.getByRole("button", { name: "Object panner" }));
    expect(screen.getByRole("dialog", { name: "Object panner" })).toHaveAttribute("aria-modal", "false");
    expect(screen.getByText("Resting position")).toBeVisible();
    expect(screen.getByText("Edits change the Supporting placement, not the front-center attention destination.")).toBeVisible();
    const title = screen.getByText("Vocals Panner");
    expect(title.previousElementSibling).toHaveStyle({ color: getStemColor("Vocals") });
    expect(screen.getByRole("dialog", { name: "Object panner" })).toHaveStyle({ borderColor: `${getStemColor("Vocals")}40` });
    expect(screen.getByLabelText("Move object panner window")).toHaveClass("cursor-default");
    expect(screen.getByLabelText("Move object panner window")).not.toHaveClass("bg-secondary");
    fireEvent.keyDown(screen.getByRole("group", { name: "Left/right and back/front" }), { key: "ArrowRight" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ azimuth_deg: expect.any(Number) }));

    const elevationPanner = screen.getByRole("group", { name: "Elevation" });
    fireEvent.keyDown(elevationPanner, { key: "ArrowUp" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ elevation_deg: 1 }));

    const elevated = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(elevated.elevation_deg).toBe(1);
    expect(elevated.left_right).toBeCloseTo(0.04, 9);
    expect(elevated.back_front).toBe(1);
  });

  it("shows the planar current position instead of button text or an icon", () => {
    const { container } = render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, azimuth_deg: 90 }} maxElevationDeg={35} onPlacement={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Object panner" })).toHaveClass("h-10", "w-10");
    expect(screen.getByRole("button", { name: "Object panner" })).toHaveTextContent("");
    const puck = container.querySelector("[data-panner-preview-puck]") as HTMLElement;
    expect(puck).toHaveStyle({ left: "0%" });
    expect(Number.parseFloat(puck.style.top)).toBeCloseTo(50, 9);
    expect(container.querySelectorAll("[data-panner-preview-channel]")).toHaveLength(0);
  });

  it("keeps the centre handle at its free Cartesian position", async () => {
    const user = userEvent.setup();
    const { container } = render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    fireEvent.keyDown(panner, { key: "ArrowDown" });
    fireEvent.keyDown(panner, { key: "ArrowDown" });

    expect(Number.parseFloat((container.ownerDocument.querySelector('[data-drag-handle="horizontal"]') as HTMLElement).style.top)).toBeCloseTo(4, 9);
  });

  it("does not jump to a stale placement echo during a drag", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    const { rerender } = render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    let captured = false;
    Object.assign(panner, {
      setPointerCapture: () => { captured = true; },
      hasPointerCapture: () => captured,
      releasePointerCapture: () => { captured = false; },
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}) }),
    });
    fireEvent.pointerDown(screen.getByRole("button", { name: "Object position" }), { pointerId: 1, button: 0, clientX: 20, clientY: 20 });
    fireEvent.pointerMove(panner, { pointerId: 1, clientX: 30, clientY: 30 });

    rerender(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={onPlacement} />);

    const handle = document.querySelector('[data-drag-handle="horizontal"]') as HTMLElement;
    expect(Number.parseFloat(handle.style.left)).toBeCloseTo(60, 9);
    expect(Number.parseFloat(handle.style.top)).toBeCloseTo(10, 9);
    fireEvent.pointerUp(panner, { pointerId: 1 });
  });

  it("drags linked channel handles through back and crossing without swapping identities", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg: 90 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    let captured = false;
    Object.assign(panner, {
      setPointerCapture: () => { captured = true; },
      hasPointerCapture: () => captured,
      releasePointerCapture: () => { captured = false; },
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}) }),
    });
    const left = screen.getByRole("button", { name: "Left channel position" });
    fireEvent.pointerDown(left, { pointerId: 2, button: 0, clientX: 14.64, clientY: 14.64 });
    expect(onPlacement).not.toHaveBeenCalled();

    fireEvent.pointerMove(panner, { pointerId: 2, clientX: 14.64, clientY: 85.36 });
    fireEvent.pointerMove(panner, { pointerId: 2, clientX: 85.36, clientY: 85.36 });

    const crossed = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(crossed.back_front).toBeLessThan(0);
    expect(crossed.width_deg).toBeCloseTo(90, 1);
    expect(screen.getByRole("button", { name: "Left channel position" })).toHaveStyle({ left: "85.35533905932738%" });
    expect(screen.getByRole("button", { name: "Right channel position" })).toHaveStyle({ left: "14.644660940672626%" });
    fireEvent.pointerUp(panner, { pointerId: 2 });
  });

  it("adjusts a focused channel without moving the anchor as an object keypress", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg: 90 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.keyDown(screen.getByRole("button", { name: "Left channel position" }), { key: "ArrowLeft" });

    const next = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(next.width_deg).toBeGreaterThan(90);
    expect(next.left_right).toBeCloseTo(0, 9);
    expect(screen.getByText(/\+\d+\u00b0/)).toBeInTheDocument();
  });

  it("leaves controls beneath the modeless window interactive", async () => {
    const user = userEvent.setup();
    const underneath = vi.fn();
    render(<><button onClick={underneath}>Underneath</button><ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={vi.fn()} /></>);

    await user.click(screen.getByRole("button", { name: "Object panner" }));
    await user.click(screen.getByRole("button", { name: "Underneath" }));

    expect(underneath).toHaveBeenCalledOnce();
    expect(screen.getByRole("dialog", { name: "Object panner" })).toBeInTheDocument();
  });

  it("moves the floating window from its title bar", async () => {
    const user = userEvent.setup();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const floatingWindow = screen.getByRole("dialog", { name: "Object panner" });
    vi.spyOn(floatingWindow, "getBoundingClientRect").mockReturnValue({
      left: 100, top: 80, right: 520, bottom: 680, width: 420, height: 600, x: 100, y: 80, toJSON: () => ({}),
    });
    fireEvent.keyDown(screen.getByLabelText("Move object panner window"), { key: "ArrowRight" });

    expect(floatingWindow).toHaveStyle({ left: "110px", top: "80px" });
  });

  it("keeps stereo width and spread when the centre handle moves", () => {
    expect(placementFromPannerPosition({ ...PLACEMENT, width_deg: 90, object_size: 0.5 }, { lateral: 0.75, depth: 0.25 })).toMatchObject({
      width_deg: 90,
      object_size: 0.5,
      elevation_deg: 0,
    });
  });

  it("updates width and size independently of the centre position", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.keyDown(screen.getByLabelText("Stereo spread"), { key: "ArrowRight" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ width_deg: 1, object_size: 0.5, azimuth_deg: 0 }));

    fireEvent.keyDown(screen.getByLabelText("Object size"), { key: "ArrowRight" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ width_deg: 1, object_size: 0.51, azimuth_deg: 0 }));
  });

  it("offers sliders for the panner's horizontal, depth, and elevation axes", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, azimuth_deg: 90, elevation_deg: 17.5 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    expect(screen.getByText("Left / Right")).toBeInTheDocument();
    expect(screen.getByText("Back / Front")).toBeInTheDocument();
    expect(screen.getAllByText("Elevation", { selector: "span" })).toHaveLength(2);
    expect(screen.getByText("-100%", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText("0%", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText("18°")).toBeInTheDocument();

    fireEvent.keyDown(screen.getByLabelText("Left to right"), { key: "ArrowRight" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ azimuth_deg: expect.any(Number) }));

    fireEvent.keyDown(screen.getByLabelText("Back to front"), { key: "ArrowLeft" });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ azimuth_deg: expect.any(Number) }));

    fireEvent.keyDown(screen.getByLabelText("Ear level to elevation"), { key: "ArrowRight" });
    expect((onPlacement.mock.calls.at(-1)?.[0] as StemPlacement).elevation_deg).toBeCloseTo(17.85, 9);
  });

  it("shows width only for linked stereo objects", async () => {
    const user = userEvent.setup();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35} objectMode="mono" onPlacement={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    expect(screen.queryByLabelText("Stereo spread")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Object size")).toBeInTheDocument();
    expect(document.querySelectorAll("[data-channel]")).toHaveLength(0);
  });

  it("resets pucks to Logic's default positions with Option-click", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, azimuth_deg: 90, elevation_deg: 20 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.pointerDown(screen.getByRole("button", { name: "Object position" }), { altKey: true });
    const front = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(front.azimuth_deg).toBeCloseTo(0, 9);
    expect(front.elevation_deg).toBe(20);

    fireEvent.pointerDown(screen.getByRole("group", { name: "Elevation" }), { altKey: true });
    expect(onPlacement).toHaveBeenLastCalledWith(expect.objectContaining({ elevation_deg: 0 }));
  });

  it("keeps direct-image and object send controls in the panner window", async () => {
    const user = userEvent.setup();
    const onObjectMode = vi.fn();
    const onAmbient = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35}
      channels={["FL", "FR", "LFE", "SL", "SR", "TFL", "TFR"]}
      ambientRear={0.5} ambientHeight={0.5} onPlacement={vi.fn()}
      onObjectMode={onObjectMode} onAmbient={onAmbient} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.change(screen.getByLabelText("Direct image"), { target: { value: "mono" } });
    fireEvent.keyDown(screen.getByLabelText("Ambience to rear"), { key: "ArrowRight" });
    fireEvent.keyDown(screen.getByLabelText("Ambience to height"), { key: "ArrowRight" });
    expect(screen.queryByLabelText("LFE send")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Center level")).not.toBeInTheDocument();

    expect(onObjectMode).toHaveBeenCalledWith("mono");
    expect(onAmbient).toHaveBeenCalledWith({ rear: 0.51 });
    expect(onAmbient).toHaveBeenCalledWith({ height: 0.51 });
  });

  it("exposes the revision-2 height cutoff separately", async () => {
    const user = userEvent.setup();
    const onAmbient = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35}
      channels={["FL", "FR", "TFL", "TFR"]} ambientHeightCutoffHz={500}
      ambientHeightCrossoverHz={4000} onPlacement={vi.fn()} onAmbient={onAmbient} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    expect(screen.getByLabelText("Height cutoff")).toBeInTheDocument();
    fireEvent.keyDown(screen.getByLabelText("Height cutoff"), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ heightCutoffHz: expect.any(Number) });
  });

  it("edits revision-2 wet trim and height texture", async () => {
    const user = userEvent.setup();
    const onAmbient = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={PLACEMENT} maxElevationDeg={35}
      channels={["FL", "FR", "SL", "SR", "TFL", "TFR"]}
      ambientTrimDb={3} heightTexture={0.1} onPlacement={vi.fn()} onAmbient={onAmbient} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.keyDown(screen.getByLabelText("Ambience trim"), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ ambientTrimDb: 3.5 });
    fireEvent.keyDown(screen.getByLabelText("Height texture"), { key: "ArrowRight" });
    expect(onAmbient).toHaveBeenLastCalledWith({ heightTexture: 0.11 });
  });

  it("places the L and R markers at the ends of the stereo image width", () => {
    const channels = objectChannelPositions({ ...PLACEMENT, width_deg: 60 });

    expect(channels.left.lateral).toBeCloseTo(0.25, 9);
    expect(channels.right.lateral).toBeCloseTo(0.75, 9);
    expect(channels.left.depth).toBeCloseTo(channels.right.depth, 9);
  });

  it("keeps the L and R markers at a freely placed centre's radius", () => {
    const channels = objectChannelPositions(
      { ...PLACEMENT, width_deg: 60 },
      { lateral: 0.5, depth: 0.4 },
    );

    expect(Math.hypot(channels.left.lateral - 0.5, channels.left.depth - 0.5)).toBeCloseTo(0.1, 9);
    expect(Math.hypot(channels.right.lateral - 0.5, channels.right.depth - 0.5)).toBeCloseTo(0.1, 9);
  });
});
