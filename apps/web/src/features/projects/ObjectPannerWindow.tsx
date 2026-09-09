import * as React from "react";
import { createPortal } from "react-dom";
import { CloudFog, MoveVertical, UserRound, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { getStemColor, getStemIcon } from "@/lib/stems";
import type { StemMovementSettings } from "@/lib/manifest";
import {
  objectChannelCoordinates,
  pannerCoordinatesFromPlacement,
  placementFromPannerCoordinates,
  type PannerCoordinates,
} from "@/lib/spatial";
import type { StemPlacement } from "./wasmEngine/panner";
import { StemMovementControls, type StemMovementDefaults } from "./StemMovementControls";

export type PannerPosition = { lateral: number; depth: number };
export {
  objectChannelCoordinates,
  pannerCoordinatesFromPlacement,
  pannerToScenePosition,
  placementFromPannerCoordinates,
} from "@/lib/spatial";

const KEY_STEP = 0.02;

function clamp(value: number) {
  return Math.min(1, Math.max(0, value));
}

function clampSigned(value: number) {
  return Math.min(1, Math.max(-1, value));
}

/** `azimuth = atan2(-x, -z)`: front is 0°, left is positive. */
export function azimuthFromPosition({ lateral, depth }: PannerPosition): number {
  const x = lateral * 2 - 1;
  const z = depth * 2 - 1;
  if (x === 0 && z === 0) return 0;
  return (Math.atan2(-x, -z) * 180) / Math.PI;
}

export function positionFromAzimuth(azimuthDeg: number): PannerPosition {
  const azimuth = (azimuthDeg * Math.PI) / 180;
  return {
    lateral: (-Math.sin(azimuth) + 1) / 2,
    depth: (-Math.cos(azimuth) + 1) / 2,
  };
}

export function pannerPositionFromPlacement(placement: StemPlacement): PannerPosition {
  const { leftRight, backFront } = pannerCoordinatesFromPlacement(placement);
  return { lateral: (leftRight + 1) / 2, depth: (1 - backFront) / 2 };
}

export function placementFromPannerPosition(placement: StemPlacement, position: PannerPosition): StemPlacement {
  return placementFromPannerCoordinates(placement, {
    leftRight: position.lateral * 2 - 1,
    backFront: 1 - position.depth * 2,
    elevation: pannerCoordinatesFromPlacement(placement).elevation,
  });
}

/** Linked-stereo feeds keep their angular width at the centre handle's radius. */
export function objectChannelPositions(placement: StemPlacement, center = pannerPositionFromPlacement(placement)) {
  const channels = objectChannelCoordinates(placementFromPannerPosition(placement, center));
  const position = ({ leftRight, backFront }: PannerCoordinates) => ({
    lateral: (leftRight + 1) / 2,
    depth: (1 - backFront) / 2,
  });
  return {
    left: position(channels.left),
    right: position(channels.right),
  };
}

function horizontalGrid() {
  return Array.from({ length: 16 }, (_, index) => (
    <span key={index} className="border-r border-b border-border/70 [&:nth-child(4n)]:border-r-0 [&:nth-last-child(-n+4)]:border-b-0" />
  ));
}

function verticalGrid() {
  return Array.from({ length: 8 }, (_, index) => (
    <span key={index} className="border-r border-b border-border/70 [&:nth-child(4n)]:border-r-0 [&:nth-last-child(-n+4)]:border-b-0" />
  ));
}

function positionFromEvent(event: React.PointerEvent<HTMLDivElement>): PannerPosition {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    lateral: clamp((event.clientX - rect.left) / rect.width),
    depth: clamp((event.clientY - rect.top) / rect.height),
  };
}

function normalizeSpread(value: number) {
  const wrapped = ((value + 180) % 360 + 360) % 360 - 180;
  return wrapped === -180 && value > 0 ? 180 : wrapped;
}

function signedDegrees(value: number) {
  const rounded = Math.round(normalizeSpread(value));
  return `${rounded > 0 ? "+" : ""}${rounded}\u00b0`;
}

type PannerDrag = {
  pointerId: number;
  target: "anchor" | "left" | "right";
  offset: PannerCoordinates;
  anchorDirection: number;
  anchorScale: number;
  startAnchor: PannerCoordinates;
  startTargetRadius: number;
  startSpread: number;
  spreadSeam: boolean;
  spread: number;
  lastAngle: number;
  lastPlacement: StemPlacement;
};

export function ObjectPannerWindow({
  stemName,
  placement,
  maxElevationDeg,
  objectMode = "linked-stereo",
  channels = [],
  ambientRear = 0,
  ambientHeight = 0,
  ambientTrimDb = 0,
  heightTexture = 0,
  ambientHeightCutoffHz = 2000,
  ariaLabel = "Object panner",
  onPlacement,
  onObjectMode = () => {},
  onAmbient = () => {},
  movement,
  movementDefaults,
  onMovement = () => {},
}: {
  stemName: string;
  placement: StemPlacement;
  maxElevationDeg: number;
  objectMode?: "linked-stereo" | "mono";
  channels?: string[];
  ambientRear?: number;
  ambientHeight?: number;
  ambientTrimDb?: number;
  heightTexture?: number;
  ambientHeightCutoffHz?: number;
  ambientHeightCrossoverHz?: number;
  ariaLabel?: string;
  onPlacement: (next: StemPlacement) => void;
  onObjectMode?: (mode: "linked-stereo" | "mono") => void;
  onAmbient?: (patch: { rear?: number; height?: number; ambientTrimDb?: number; heightTexture?: number; heightCrossoverHz?: number; heightCutoffHz?: number }) => void;
  movement?: StemMovementSettings;
  movementDefaults?: StemMovementDefaults;
  onMovement?: (value: StemMovementSettings) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const [windowPosition, setWindowPosition] = React.useState<{ left: number; top: number } | null>(null);
  const windowRef = React.useRef<HTMLDivElement>(null);
  const dragRef = React.useRef<{ pointerId: number; offsetX: number; offsetY: number } | null>(null);
  const pannerDragRef = React.useRef<PannerDrag | null>(null);
  const lastChannelRef = React.useRef<"left" | "right">("right");
  const pannerDraggingRef = React.useRef(false);
  const [localPlacement, setLocalPlacement] = React.useState(placement);
  const position = pannerPositionFromPlacement(localPlacement);
  const elevation = maxElevationDeg ? clamp(localPlacement.elevation_deg / maxElevationDeg) : 0;
  const stereo = objectMode === "linked-stereo";
  const hasSurround = channels.includes("SL") || channels.includes("SR") || channels.includes("BL") || channels.includes("BR");
  const hasHeight = channels.includes("TFL") || channels.includes("TFR") || channels.includes("TBL") || channels.includes("TBR");
  const heightTone = Math.min(4000, Math.max(500, ambientHeightCutoffHz));
  const heightTonePosition = Math.log(heightTone / 500) / Math.log(8);
  const heightToneLabel = "Height cutoff";
  const StemIcon = getStemIcon(stemName);
  const stemColor = getStemColor(stemName);
  const channelPositions = objectChannelPositions(localPlacement);
  const commitPlacement = (next: StemPlacement) => {
    setLocalPlacement(next);
    onPlacement(next);
  };
  const setPosition = (next: PannerPosition) => {
    commitPlacement(placementFromPannerPosition(localPlacement, next));
  };
  const setElevationPosition = (lateral: number, nextElevation: number) => {
    const nextPosition = { lateral, depth: position.depth };
    const nextPlacement = {
      ...placementFromPannerPosition(localPlacement, nextPosition),
      elevation_deg: clamp(nextElevation) * maxElevationDeg,
    };
    commitPlacement(nextPlacement);
  };
  const moveElevationPosition = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setElevationPosition(
      position.lateral,
      clamp(1 - (event.clientY - rect.top) / rect.height),
    );
  };
  const moveChannelByKey = (channel: "left" | "right", event: React.KeyboardEvent<HTMLButtonElement>) => {
    const delta = event.shiftKey ? 0.1 : KEY_STEP;
    const target = { ...channelPositions[channel] };
    if (event.key === "ArrowLeft") target.lateral -= delta;
    else if (event.key === "ArrowRight") target.lateral += delta;
    else if (event.key === "ArrowUp") target.depth -= delta;
    else if (event.key === "ArrowDown") target.depth += delta;
    else return;
    event.preventDefault();
    event.stopPropagation();
    const canonical = {
      leftRight: clampSigned(target.lateral * 2 - 1),
      backFront: clampSigned(1 - target.depth * 2),
    };
    const current = objectChannelCoordinates(localPlacement)[channel];
    let angleDelta = Math.atan2(canonical.leftRight, canonical.backFront)
      - Math.atan2(current.leftRight, current.backFront);
    if (angleDelta > Math.PI) angleDelta -= 2 * Math.PI;
    if (angleDelta < -Math.PI) angleDelta += 2 * Math.PI;
    const width_deg = normalizeSpread(localPlacement.width_deg
      + (channel === "left" ? -2 : 2) * angleDelta * 180 / Math.PI);
    const anchor = pannerCoordinatesFromPlacement(localPlacement);
    const currentRadius = Math.hypot(current.leftRight, current.backFront);
    const anchorScale = Math.hypot(anchor.leftRight, anchor.backFront) / Math.max(
      1e-6,
      currentRadius * Math.abs(Math.cos(localPlacement.width_deg * Math.PI / 360)),
    );
    const radius = Math.hypot(canonical.leftRight, canonical.backFront) * anchorScale
      * Math.abs(Math.cos(width_deg * Math.PI / 360));
    const direction = Math.atan2(anchor.leftRight, anchor.backFront);
    commitPlacement({
      ...placementFromPannerCoordinates(localPlacement, {
        ...anchor,
        leftRight: radius * Math.sin(direction),
        backFront: radius * Math.cos(direction),
      }),
      width_deg,
    });
  };

  React.useEffect(() => {
    if (open) windowRef.current?.focus();
  }, [open]);

  React.useEffect(() => {
    if (pannerDraggingRef.current) return;
    setLocalPlacement(placement);
  }, [placement]);

  const placeWindow = (left: number, top: number) => {
    const rect = windowRef.current?.getBoundingClientRect();
    if (!rect) return;
    setWindowPosition({
      left: Math.min(window.innerWidth - 64, Math.max(64 - rect.width, left)),
      top: Math.min(window.innerHeight - 32, Math.max(0, top)),
    });
  };

  const floatingWindow = open && createPortal(
    <div className="pointer-events-none fixed inset-0 z-50">
      <div
        ref={windowRef}
        role="dialog"
        aria-modal="false"
        aria-label={ariaLabel}
        tabIndex={-1}
        className="pointer-events-auto fixed max-h-[calc(100vh-24px)] w-[420px] max-w-[calc(100vw-24px)] overflow-auto rounded-lg border bg-card shadow-2xl outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
        style={{
          borderColor: `${stemColor}40`,
          ...(windowPosition
            ? { left: windowPosition.left, top: windowPosition.top }
            : { left: "50%", top: "50%", transform: "translate(-50%, -50%)" }),
        }}
        onKeyDown={(event) => { if (event.key === "Escape") setOpen(false); }}
      >
        <div
          tabIndex={0}
          aria-label="Move object panner window"
          className="flex h-8 touch-none cursor-default items-center border-b px-3 text-[12px] font-semibold outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60"
          onPointerDown={(event) => {
            if (event.button !== 0 || !windowRef.current) return;
            const rect = windowRef.current.getBoundingClientRect();
            dragRef.current = { pointerId: event.pointerId, offsetX: event.clientX - rect.left, offsetY: event.clientY - rect.top };
            event.currentTarget.setPointerCapture(event.pointerId);
            setWindowPosition({ left: rect.left, top: rect.top });
          }}
          onPointerMove={(event) => {
            const drag = dragRef.current;
            if (drag?.pointerId === event.pointerId) placeWindow(event.clientX - drag.offsetX, event.clientY - drag.offsetY);
          }}
          onPointerUp={(event) => {
            if (dragRef.current?.pointerId !== event.pointerId) return;
            dragRef.current = null;
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onKeyDown={(event) => {
            if (!event.key.startsWith("Arrow") || !windowRef.current) return;
            const rect = windowRef.current.getBoundingClientRect();
            const step = event.shiftKey ? 40 : 10;
            event.preventDefault();
            placeWindow(
              rect.left + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0),
              rect.top + (event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0),
            );
          }}
        >
          <StemIcon className="mr-1.5 h-3.5 w-3.5 shrink-0" style={{ color: stemColor }} aria-hidden="true" />
          <span className="min-w-0 flex-1 truncate">{stemName} Panner</span>
          <button
            type="button"
            aria-label="Close object panner"
            className="-mr-1 rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
            onPointerDown={(event) => event.stopPropagation()}
            onClick={() => setOpen(false)}
          ><X className="h-3.5 w-3.5" /></button>
        </div>
        <div className="flex flex-col items-center gap-6 px-6 py-5">
          <div className="w-full text-[11px] text-muted-foreground">
            <p className="font-semibold text-foreground">Resting position</p>
            <p>Edits change the Supporting placement, not the front-center attention destination.</p>
          </div>
          <section className="w-full" aria-label="Left/right and back/front position">
            <div className="mb-1 grid grid-cols-[44px_minmax(0,1fr)_44px] text-[12px] text-muted-foreground"><span className="col-start-2 text-center">Front</span></div>
            <div className="grid grid-cols-[44px_minmax(0,1fr)_44px] items-center text-[12px] text-muted-foreground">
              <span className="pr-3 text-right">Left</span>
              <div
                role="group"
                tabIndex={0}
                aria-label="Left/right and back/front"
                aria-description="Drag the centre puck or use arrow keys to pan left, right, front, and back. Option-click resets to centre front."
                className="relative aspect-square touch-none bg-muted/50 outline-none ring-1 ring-border focus-visible:ring-2 focus-visible:ring-ring/60"
                onPointerDown={(event) => {
                  const element = (event.target as HTMLElement).closest<HTMLElement>("[data-panner-target]");
                  if (!element || event.button !== 0) return;
                  if (event.altKey) { setPosition({ lateral: 0.5, depth: 0 }); return; }
                  let target = element.dataset.pannerTarget as PannerDrag["target"];
                  const overlap = Math.hypot(
                    channelPositions.left.lateral - channelPositions.right.lateral,
                    channelPositions.left.depth - channelPositions.right.depth,
                  ) < 1e-6;
                  if (target !== "anchor" && overlap) target = lastChannelRef.current === "left" ? "right" : "left";
                  if (target !== "anchor") lastChannelRef.current = target;
                  const pointer = positionFromEvent(event);
                  const pointerCoordinates = {
                    leftRight: pointer.lateral * 2 - 1,
                    backFront: 1 - pointer.depth * 2,
                    elevation: pannerCoordinatesFromPlacement(localPlacement).elevation,
                  };
                  const targetPosition = target === "anchor"
                    ? pannerCoordinatesFromPlacement(localPlacement)
                    : objectChannelCoordinates(localPlacement)[target];
                  pannerDraggingRef.current = true;
                  event.currentTarget.setPointerCapture(event.pointerId);
                  event.preventDefault();
                  const anchor = pannerCoordinatesFromPlacement(localPlacement);
                  const targetRadius = Math.hypot(targetPosition.leftRight, targetPosition.backFront);
                  const spreadCosine = Math.abs(Math.cos(localPlacement.width_deg * Math.PI / 360));
                  pannerDragRef.current = {
                    pointerId: event.pointerId,
                    target,
                    offset: {
                      leftRight: pointerCoordinates.leftRight - targetPosition.leftRight,
                      backFront: pointerCoordinates.backFront - targetPosition.backFront,
                      elevation: 0,
                    },
                    anchorDirection: Math.atan2(
                      anchor.leftRight,
                      anchor.backFront,
                    ),
                    anchorScale: spreadCosine < 1e-6 ? 1 : Math.hypot(anchor.leftRight, anchor.backFront)
                      / Math.max(1e-6, targetRadius * spreadCosine),
                    startAnchor: anchor,
                    startTargetRadius: targetRadius,
                    startSpread: localPlacement.width_deg,
                    spreadSeam: spreadCosine < 1e-6,
                    spread: localPlacement.width_deg,
                    lastAngle: Math.atan2(targetPosition.leftRight, targetPosition.backFront) * 180 / Math.PI,
                    lastPlacement: localPlacement,
                  };
                }}
                onPointerMove={(event) => {
                  const drag = pannerDragRef.current;
                  if (!drag || drag.pointerId !== event.pointerId || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
                  event.preventDefault();
                  const pointer = positionFromEvent(event);
                  const target = {
                    leftRight: clampSigned(pointer.lateral * 2 - 1 - drag.offset.leftRight),
                    backFront: clampSigned(1 - pointer.depth * 2 - drag.offset.backFront),
                    elevation: pannerCoordinatesFromPlacement(drag.lastPlacement).elevation,
                  };
                  if (drag.target === "anchor") {
                    const next = placementFromPannerCoordinates(drag.lastPlacement, target);
                    drag.lastPlacement = next;
                    commitPlacement(next);
                    return;
                  }
                  let angle = Math.atan2(target.leftRight, target.backFront) * 180 / Math.PI;
                  while (angle - drag.lastAngle > 180) angle -= 360;
                  while (angle - drag.lastAngle < -180) angle += 360;
                  drag.lastAngle = angle;
                  const radius = Math.hypot(target.leftRight, target.backFront);
                  const radialDelta = radius - drag.startTargetRadius;
                  const provisionalAnchor = radius > 1e-6 ? {
                    leftRight: drag.startAnchor.leftRight + radialDelta * target.leftRight / radius,
                    backFront: drag.startAnchor.backFront + radialDelta * target.backFront / radius,
                  } : drag.startAnchor;
                  const provisionalDirection = Math.hypot(
                    provisionalAnchor.leftRight,
                    provisionalAnchor.backFront,
                  ) > 1e-6
                    ? Math.atan2(provisionalAnchor.leftRight, provisionalAnchor.backFront)
                    : drag.anchorDirection;
                  let spread = (drag.target === "left" ? -2 : 2)
                    * (angle - provisionalDirection * 180 / Math.PI);
                  while (spread - drag.spread > 360) spread -= 720;
                  while (spread - drag.spread < -360) spread += 720;
                  drag.spread = spread;
                  if (drag.spreadSeam && Math.abs(drag.spread - drag.startSpread) >= 180) {
                    drag.spreadSeam = false;
                  }
                  const normalizedSpread = normalizeSpread(drag.spread);
                  const width_deg = drag.spreadSeam ? drag.spread : normalizedSpread;
                  const turns = Math.round((drag.spread - normalizedSpread) / 360);
                  const anchorDirection = provisionalDirection + turns * Math.PI;
                  // Collapse through the listener before switching the signed
                  // spread representation, preserving the same channel points.
                  const anchorRadius = radius * drag.anchorScale
                    * Math.abs(Math.cos(width_deg * Math.PI / 360));
                  const geometricAnchor = {
                    leftRight: anchorRadius * Math.sin(anchorDirection),
                    backFront: anchorRadius * Math.cos(anchorDirection),
                  };
                  const seamProgress = drag.spreadSeam
                    ? Math.min(1, Math.abs(drag.spread - drag.startSpread) / 180 + Math.abs(radialDelta))
                    : 1;
                  const anchor = {
                    leftRight: drag.startAnchor.leftRight * (1 - seamProgress) + geometricAnchor.leftRight * seamProgress,
                    backFront: drag.startAnchor.backFront * (1 - seamProgress) + geometricAnchor.backFront * seamProgress,
                    elevation: target.elevation,
                  };
                  const next = { ...placementFromPannerCoordinates(drag.lastPlacement, anchor), width_deg };
                  drag.lastPlacement = next;
                  commitPlacement(next);
                }}
                onPointerUp={(event) => {
                  const drag = pannerDragRef.current;
                  if (!drag || drag.pointerId !== event.pointerId) return;
                  if (drag.target !== "anchor") commitPlacement({
                    ...drag.lastPlacement,
                    width_deg: drag.spreadSeam ? drag.spread : normalizeSpread(drag.spread),
                  });
                  pannerDragRef.current = null;
                  pannerDraggingRef.current = false;
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }}
                onPointerCancel={(event) => {
                  if (pannerDragRef.current?.pointerId !== event.pointerId) return;
                  pannerDragRef.current = null;
                  pannerDraggingRef.current = false;
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
                }}
                onKeyDown={(event) => {
                  const next = { ...position };
                  if (event.key === "ArrowLeft") next.lateral -= KEY_STEP;
                  else if (event.key === "ArrowRight") next.lateral += KEY_STEP;
                  else if (event.key === "ArrowUp") next.depth -= KEY_STEP;
                  else if (event.key === "ArrowDown") next.depth += KEY_STEP;
                  else return;
                  event.preventDefault();
                  setPosition({ lateral: clamp(next.lateral), depth: clamp(next.depth) });
                }}
              >
                <div aria-hidden="true" className="absolute inset-0 grid grid-cols-4 grid-rows-4">{horizontalGrid()}</div>
                {stereo && <><button type="button" aria-label="Left channel position" data-channel="left" data-panner-target="left" className="absolute z-10 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border bg-secondary text-[11px] font-semibold text-foreground outline-none hover:border-primary active:bg-accent focus-visible:ring-2 focus-visible:ring-ring" style={{ left: `${channelPositions.left.lateral * 100}%`, top: `${channelPositions.left.depth * 100}%` }} onKeyDown={(event) => moveChannelByKey("left", event)}>L</button>
                <button type="button" aria-label="Right channel position" data-channel="right" data-panner-target="right" className="absolute z-10 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border bg-secondary text-[11px] font-semibold text-foreground outline-none hover:border-primary active:bg-accent focus-visible:ring-2 focus-visible:ring-ring" style={{ left: `${channelPositions.right.lateral * 100}%`, top: `${channelPositions.right.depth * 100}%` }} onKeyDown={(event) => moveChannelByKey("right", event)}>R</button></>}
                <UserRound aria-hidden="true" className="pointer-events-none absolute left-1/2 top-1/2 h-9 w-9 -translate-x-1/2 -translate-y-1/2 text-muted-foreground/60" />
                <button
                  type="button"
                  aria-label="Object position"
                  data-panner-target="anchor"
                  data-drag-handle="horizontal"
                  className="absolute z-20 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-card shadow-sm outline-none ring-4 ring-primary/20 hover:bg-accent active:scale-95 focus-visible:ring-4 focus-visible:ring-ring"
                  style={{ left: `${position.lateral * 100}%`, top: `${position.depth * 100}%` }}
                />
              </div>
              <span className="pl-3">Right</span>
            </div>
            <div className="mt-1 grid grid-cols-[44px_minmax(0,1fr)_44px] text-[12px] text-muted-foreground"><span className="col-start-2 text-center">Back</span></div>
          </section>
          {maxElevationDeg > 0 && <section className="w-full" aria-label="Elevation">
            <div className="grid grid-cols-[44px_minmax(0,1fr)_44px] items-stretch text-[12px] text-muted-foreground">
              <div
                role="group"
                tabIndex={0}
                aria-label="Elevation"
                aria-description="Drag vertically or use up and down arrow keys to set elevation. Option-click resets to ear level."
                className="relative col-start-2 h-40 touch-none bg-muted/50 outline-none ring-1 ring-border focus-visible:ring-2 focus-visible:ring-ring/60"
                onPointerDown={(event) => {
                  if (event.altKey) { setElevationPosition(position.lateral, 0); return; }
                  pannerDraggingRef.current = true;
                  event.currentTarget.setPointerCapture(event.pointerId);
                  moveElevationPosition(event);
                }}
                onPointerMove={(event) => {
                  if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
                  moveElevationPosition(event);
                }}
                onPointerUp={(event) => {
                  pannerDraggingRef.current = false;
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
                }}
                onPointerCancel={(event) => {
                  pannerDraggingRef.current = false;
                  if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
                }}
                onKeyDown={(event) => {
                  let nextElevation = elevation;
                  if (event.key === "ArrowUp") nextElevation += 1 / maxElevationDeg;
                  else if (event.key === "ArrowDown") nextElevation -= 1 / maxElevationDeg;
                  else return;
                  event.preventDefault();
                  setElevationPosition(position.lateral, clamp(nextElevation));
                }}
              >
                <div aria-hidden="true" className="absolute inset-0 grid grid-cols-4 grid-rows-2">{verticalGrid()}</div>
                <UserRound aria-hidden="true" className="pointer-events-none absolute bottom-0 left-1/2 h-9 w-9 -translate-x-1/2 text-muted-foreground/60" />
                {stereo && <><span className="pointer-events-none absolute z-10 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border bg-secondary text-[11px] font-semibold text-foreground" style={{ left: `${channelPositions.left.lateral * 100}%`, top: `${(1 - elevation) * 100}%` }}>L</span>
                <span className="pointer-events-none absolute z-10 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border bg-secondary text-[11px] font-semibold text-foreground" style={{ left: `${channelPositions.right.lateral * 100}%`, top: `${(1 - elevation) * 100}%` }}>R</span></>}
                <span
                  data-drag-handle="elevation"
                  className="pointer-events-none absolute z-20 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-card shadow-sm ring-4 ring-primary/20"
                  style={{ left: `${position.lateral * 100}%`, top: `${(1 - elevation) * 100}%` }}
                />
              </div>
              <div className="col-start-3 flex flex-col justify-between py-0.5 pl-3"><span>Elevation</span><span>Ear level</span></div>
            </div>
          </section>}
          <div className="grid w-full gap-3 border-t pt-4 sm:grid-cols-3">
            <label className="block text-[11px] text-muted-foreground">
              <span className="flex"><span>Left / Right</span><span className="ml-auto tabular-nums">{Math.round((position.lateral * 2 - 1) * 100)}%</span></span>
              <Slider aria-label="Left to right" className="mt-1.5" min={0} max={1} step={0.01}
                value={[position.lateral]} onValueChange={([lateral]) => setPosition({ ...position, lateral })} />
            </label>
            <label className="block text-[11px] text-muted-foreground">
              <span className="flex"><span>Back / Front</span><span className="ml-auto tabular-nums">{Math.round((1 - position.depth * 2) * 100)}%</span></span>
              <Slider aria-label="Back to front" className="mt-1.5" min={0} max={1} step={0.01}
                value={[1 - position.depth]} onValueChange={([front]) => setPosition({ ...position, depth: 1 - front })} />
            </label>
            {maxElevationDeg > 0 && <label className="block text-[11px] text-muted-foreground">
              <span className="flex"><span>Elevation</span><span className="ml-auto tabular-nums">{Math.round(elevation * maxElevationDeg)}°</span></span>
              <Slider aria-label="Ear level to elevation" className="mt-1.5" min={0} max={1} step={0.01}
                value={[elevation]} onValueChange={([nextElevation]) => setElevationPosition(position.lateral, nextElevation)} />
            </label>}
          </div>
          <div className="w-full space-y-3 border-t pt-4">
            <label className="block text-[11px] text-muted-foreground">
              <span>Direct image</span>
              <select aria-label="Direct image" className="mt-1.5 flex h-7 w-full rounded-md border bg-secondary px-2 text-[13px] text-foreground"
                value={objectMode} onChange={(event) => onObjectMode(event.target.value as "linked-stereo" | "mono")}>
                <option value="linked-stereo">Linked stereo</option><option value="mono">Mono</option>
              </select>
            </label>
            <div className={`grid gap-3${stereo ? " sm:grid-cols-2" : ""}`}>
              {stereo && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center"><span>Spread</span><span className="ml-auto tabular-nums">{signedDegrees(localPlacement.width_deg)}</span></span>
                <Slider aria-label="Stereo spread" className="mt-1.5" min={-180} max={180} step={1}
                  value={[normalizeSpread(localPlacement.width_deg)]} onValueChange={([width_deg]) => commitPlacement({ ...localPlacement, width_deg })} />
              </label>}
              <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center"><span>Size</span><span className="ml-auto tabular-nums">{Math.round(localPlacement.object_size * 100)}%</span></span>
                <Slider aria-label="Object size" className="mt-1.5" min={0} max={1} step={0.01}
                  value={[localPlacement.object_size]} onValueChange={([object_size]) => commitPlacement({ ...localPlacement, object_size })} />
              </label>
            </div>
            {(hasSurround || hasHeight) && <div className="grid gap-3 sm:grid-cols-2">
              {hasSurround && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1"><CloudFog className="h-3 w-3" />Ambience to rear</span>
                <Slider aria-label="Ambience to rear" className="mt-1.5" min={0} max={1} step={0.01}
                  value={[ambientRear]} onValueChange={([rear]) => onAmbient({ rear })} />
              </label>}
              {hasHeight && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1"><CloudFog className="h-3 w-3" />Ambience to height</span>
                <Slider aria-label="Ambience to height" className="mt-1.5" min={0} max={1} step={0.01}
                  value={[ambientHeight]} onValueChange={([height]) => onAmbient({ height })} />
              </label>}
              {hasHeight && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1"><MoveVertical className="h-3 w-3" />{heightToneLabel} <span className="ml-auto">{Math.round(heightTone)} Hz</span></span>
                <Slider aria-label={heightToneLabel} className="mt-1.5" min={0} max={1} step={0.01}
                  value={[heightTonePosition]} onValueChange={([value]) => onAmbient({ heightCutoffHz: 500 * 8 ** value })} />
              </label>}
              {(hasSurround || hasHeight) && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1"><CloudFog className="h-3 w-3" />Ambience trim <span className="ml-auto tabular-nums">{ambientTrimDb > 0 ? "+" : ""}{ambientTrimDb.toFixed(1)} dB</span></span>
                <Slider aria-label="Ambience trim" className="mt-1.5" min={0} max={6} step={0.5}
                  value={[ambientTrimDb]} onValueChange={([trim]) => onAmbient({ ambientTrimDb: trim })} />
              </label>}
              {hasHeight && <label className="block text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1"><MoveVertical className="h-3 w-3" />Height texture <span className="ml-auto tabular-nums">{Math.round(heightTexture * 100)}%</span></span>
                <Slider aria-label="Height texture" className="mt-1.5" min={0} max={0.25} step={0.01}
                  value={[heightTexture]} onValueChange={([texture]) => onAmbient({ heightTexture: texture })} />
              </label>}
            </div>}
          </div>
          <StemMovementControls stemName={stemName} value={movement} channels={channels} defaults={movementDefaults} onChange={onMovement} />
        </div>
      </div>
    </div>,
    document.body,
  );

  return (
    <>
      <Button
        className="relative h-10 w-10 overflow-hidden p-0"
        variant="outline"
        size="icon"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={ariaLabel}
        onClick={() => setOpen(true)}
      >
        <span data-panner-preview className="relative h-7 w-7 border border-border/70 bg-muted/50">
          <span aria-hidden="true" className="absolute inset-x-1/2 top-0 h-px -translate-x-1/2 bg-border" />
          <span aria-hidden="true" className="absolute inset-y-1/2 left-0 w-px -translate-y-1/2 bg-border" />
          <span aria-hidden="true" className="absolute inset-y-1/2 right-0 w-px -translate-y-1/2 bg-border" />
          <span data-panner-preview-puck className="absolute z-20 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-primary bg-card ring-2 ring-primary/20" style={{ left: `${position.lateral * 100}%`, top: `${position.depth * 100}%` }} />
        </span>
      </Button>
      {floatingWindow}
    </>
  );
}
