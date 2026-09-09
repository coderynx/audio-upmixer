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
    expect(channels.left.leftRight).toBeCloseTo(-1, 9);
    expect(channels.right.leftRight).toBeCloseTo(1, 9);
    expect(channels.left.backFront).toBeCloseTo(1, 9);
    expect(channels.right.backFront).toBeCloseTo(1, 9);
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

  it("moves both rendered channels only vertically for elevation and never for size", () => {
    const base: StemPlacement = { ...PLACEMENT, left_right: 0.25, back_front: 0.75, width_deg: 60 };
    const sizedPlacement = { ...base, object_size: 1 };
    const flat = objectChannelCoordinates(base);
    const elevated = objectChannelCoordinates({ ...base, elevation_deg: 30 });
    const sized = objectChannelCoordinates(sizedPlacement);

    expect(elevated.left.leftRight).toBe(flat.left.leftRight);
    expect(elevated.left.backFront).toBe(flat.left.backFront);
    expect(elevated.right.leftRight).toBe(flat.right.leftRight);
    expect(elevated.right.backFront).toBe(flat.right.backFront);
    expect(elevated.left.elevation).toBeCloseTo(0.5, 12);
    expect(elevated.right.elevation).toBeCloseTo(0.5, 12);
    expect(sized).toEqual(flat);
  });

  it("keeps channel labels attached after signed spread crosses", () => {
    const placement = { ...PLACEMENT, left_right: 0, back_front: 1, width_deg: -90 };
    const channels = objectChannelCoordinates(placement);

    expect(channels.left.leftRight).toBeGreaterThan(0);
    expect(channels.right.leftRight).toBeLessThan(0);
    expect(objectChannelCoordinates(JSON.parse(JSON.stringify(placement)))).toEqual(channels);
  });

  it("keeps anchor and channel positions continuous across signed wrap", () => {
    const seam = Math.cos(89.5 * Math.PI / 180);
    const before = { ...PLACEMENT, left_right: 0, back_front: seam, width_deg: 179 };
    const after = { ...PLACEMENT, left_right: 0, back_front: -seam, width_deg: -179 };
    const beforeChannels = objectChannelCoordinates(before);
    const afterChannels = objectChannelCoordinates(after);

    expect(Math.abs(before.back_front - after.back_front)).toBeLessThan(0.02);
    expect(Math.hypot(
      beforeChannels.left.leftRight - afterChannels.left.leftRight,
      beforeChannels.left.backFront - afterChannels.left.backFront,
    )).toBeLessThan(0.02);
    expect(Math.hypot(
      beforeChannels.right.leftRight - afterChannels.right.leftRight,
      beforeChannels.right.backFront - afterChannels.right.backFront,
    )).toBeLessThan(0.02);
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
    fireEvent.pointerDown(left, { pointerId: 2, button: 0, clientX: 0, clientY: 0 });
    expect(onPlacement).not.toHaveBeenCalled();

    fireEvent.pointerMove(panner, { pointerId: 2, clientX: 0, clientY: 100 });
    fireEvent.pointerMove(panner, { pointerId: 2, clientX: 100, clientY: 100 });

    const crossed = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(crossed.back_front).toBeLessThan(0);
    expect(crossed.width_deg).toBeCloseTo(90, 1);
    const leftPosition = Number.parseFloat(screen.getByRole("button", { name: "Left channel position" }).style.left);
    const rightPosition = Number.parseFloat(screen.getByRole("button", { name: "Right channel position" }).style.left);
    expect(leftPosition).toBeGreaterThan(rightPosition);
    fireEvent.pointerUp(panner, { pointerId: 2 });
  });

  it("returns the Right channel to its identity after a complete orbit", async () => {
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
    fireEvent.pointerDown(screen.getByRole("button", { name: "Right channel position" }), { pointerId: 3, button: 0, clientX: 100, clientY: 0 });
    for (const [clientX, clientY] of [[100, 100], [0, 100], [0, 0], [100, 0]]) {
      fireEvent.pointerMove(panner, { pointerId: 3, clientX, clientY });
    }
    fireEvent.pointerUp(panner, { pointerId: 3 });

    const final = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(final.left_right).toBeCloseTo(0, 12);
    expect(final.back_front).toBeCloseTo(1, 12);
    expect(final.width_deg).toBe(90);
    expect(screen.getByRole("button", { name: "Left channel position" })).toHaveTextContent("L");
    expect(screen.getByRole("button", { name: "Right channel position" })).toHaveTextContent("R");
  });

  it("moves the anchor laterally for an asymmetric channel drag", async () => {
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
    fireEvent.pointerDown(screen.getByRole("button", { name: "Left channel position" }), { pointerId: 4, button: 0, clientX: 0, clientY: 0 });
    fireEvent.pointerMove(panner, { pointerId: 4, clientX: 0, clientY: 25 });

    const next = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(next.left_right).not.toBeCloseTo(0, 6);
    expect(next.back_front).not.toBeCloseTo(1, 6);
    const leftPosition = screen.getByRole("button", { name: "Left channel position" });
    expect(Number.parseFloat(leftPosition.style.left)).toBeCloseTo(0, 9);
    expect(Number.parseFloat(leftPosition.style.top)).toBeCloseTo(25, 9);
    expect(screen.getByRole("button", { name: "Right channel position" }).style.top).not.toBe("0%");
  });

  it.each([180, -180])("starts a %s degree handle drag without teleporting", async (width_deg) => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    let captured = false;
    Object.assign(panner, {
      setPointerCapture: () => { captured = true; },
      hasPointerCapture: () => captured,
      releasePointerCapture: () => { captured = false; },
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}) }),
    });
    const clientX = width_deg > 0 ? 0 : 100;
    fireEvent.pointerDown(screen.getByRole("button", { name: "Left channel position" }), { pointerId: 5, button: 0, clientX, clientY: 50 });
    fireEvent.pointerMove(panner, { pointerId: 5, clientX, clientY: 51 });

    const next = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(next.back_front).toBeGreaterThan(0.9);
    expect(Math.abs(next.left_right ?? 0)).toBeLessThan(0.1);
    const left = screen.getByRole("button", { name: "Left channel position" });
    expect(Number.parseFloat(left.style.left)).toBeCloseTo(clientX, 0);
    expect(Number.parseFloat(left.style.top)).toBeCloseTo(51, 0);
  });

  it.each([180, -180])("keeps a continued orbit from %s degrees within persisted bounds", async (width_deg) => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    let captured = false;
    Object.assign(panner, {
      setPointerCapture: () => { captured = true; },
      hasPointerCapture: () => captured,
      releasePointerCapture: () => { captured = false; },
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}) }),
    });
    const clientX = width_deg > 0 ? 0 : 100;
    const orbit = width_deg > 0
      ? [[0, 100], [100, 100], [100, 0], [0, 0], [0, 50]]
      : [[100, 100], [0, 100], [0, 0], [100, 0], [100, 50]];
    fireEvent.pointerDown(screen.getByRole("button", { name: "Left channel position" }), { pointerId: 6, button: 0, clientX, clientY: 50 });
    for (const [x, y] of orbit) fireEvent.pointerMove(panner, { pointerId: 6, clientX: x, clientY: y });
    fireEvent.pointerUp(panner, { pointerId: 6 });

    expect(onPlacement.mock.calls.every(([next]) => Math.abs(next.width_deg) <= 360)).toBe(true);
    expect(screen.getByRole("button", { name: "Left channel position" })).toHaveTextContent("L");
    expect(screen.getByRole("button", { name: "Right channel position" })).toHaveTextContent("R");
  });

  it("keeps the anchor continuous when an exact-seam drag becomes normalized", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg: 180 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    const panner = screen.getByRole("group", { name: "Left/right and back/front" });
    Object.assign(panner, {
      setPointerCapture: () => {},
      hasPointerCapture: () => true,
      releasePointerCapture: () => {},
      getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}) }),
    });
    fireEvent.pointerDown(screen.getByRole("button", { name: "Left channel position" }), { pointerId: 7, button: 0, clientX: 0, clientY: 50 });
    const point = (degrees: number) => ({
      clientX: 50 + 50 * Math.sin(degrees * Math.PI / 180),
      clientY: 50 - 50 * Math.cos(degrees * Math.PI / 180),
    });
    fireEvent.pointerMove(panner, { pointerId: 7, ...point(-179) });
    const before = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    fireEvent.pointerMove(panner, { pointerId: 7, ...point(-181) });
    const after = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;

    expect(Math.hypot(
      (after.left_right ?? 0) - (before.left_right ?? 0),
      (after.back_front ?? 0) - (before.back_front ?? 0),
    )).toBeLessThan(0.05);
  });

  it("adjusts a focused channel as a linked object gesture", async () => {
    const user = userEvent.setup();
    const onPlacement = vi.fn();
    render(<ObjectPannerWindow stemName="Vocals" placement={{ ...PLACEMENT, left_right: 0, back_front: 1, width_deg: 90 }} maxElevationDeg={35} onPlacement={onPlacement} />);
    await user.click(screen.getByRole("button", { name: "Object panner" }));

    fireEvent.keyDown(screen.getByRole("button", { name: "Left channel position" }), { key: "ArrowDown" });

    const next = onPlacement.mock.calls.at(-1)?.[0] as StemPlacement;
    expect(next.width_deg).not.toBe(90);
    expect(next.left_right).toBeCloseTo(0, 9);
    expect(next.back_front).not.toBe(1);
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

    expect(channels.left.lateral).toBeCloseTo(0.2113248654, 9);
    expect(channels.right.lateral).toBeCloseTo(0.7886751346, 9);
    expect(channels.left.depth).toBeCloseTo(channels.right.depth, 9);
  });

  it("expands channel radius with spread around a freely placed centre", () => {
    const channels = objectChannelPositions(
      { ...PLACEMENT, width_deg: 60 },
      { lateral: 0.5, depth: 0.4 },
    );

    expect(Math.hypot(channels.left.lateral - 0.5, channels.left.depth - 0.5)).toBeCloseTo(0.1154700538, 9);
    expect(Math.hypot(channels.right.lateral - 0.5, channels.right.depth - 0.5)).toBeCloseTo(0.1154700538, 9);
  });
});
