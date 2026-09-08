import type {
  MovementEvent,
  MovementSchedule,
  MovementStemSchedule,
} from "./engineTypes";

/** The shared compiler evaluates events on absolute programme time. Keep this
 * tiny reader in the delivery layer so the scene can show the same immutable
 * schedule without creating or saving animated placement state. */
export function movementStemFor(
  schedule: MovementSchedule | null | undefined,
  stemKey: string,
): MovementStemSchedule | undefined {
  if (!schedule) return undefined;
  return schedule.stems.find((stem) => stem.stem_key === stemKey);
}

function eventAt(events: MovementEvent[], timeUs: number): {
  current: MovementEvent;
  previous: MovementEvent;
} | undefined {
  if (!events.length) return undefined;
  const time = Math.max(0, timeUs);
  // Schedules can span a full programme. Keep scene scrubbing O(log n) and
  // avoid allocating a sliced event array on every animation frame.
  let low = 0;
  let high = events.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (events[middle].time_us <= time) low = middle + 1;
    else high = middle;
  }
  const currentIndex = Math.max(0, low - 1);
  return {
    current: events[currentIndex],
    previous: events[Math.max(0, currentIndex - 1)],
  };
}

function interpolate(previous: number[], current: number[], amount: number): number[] {
  return current.map((value, index) =>
    previous[index] + (value - previous[index]) * amount,
  );
}

function interpolatePosition(
  previous: [number, number, number],
  current: [number, number, number],
  amount: number,
): [number, number, number] {
  return [
    previous[0] + (current[0] - previous[0]) * amount,
    previous[1] + (current[1] - previous[1]) * amount,
    previous[2] + (current[2] - previous[2]) * amount,
  ];
}

/** Evaluate an event endpoint using the shared 5.208 ms transition. */
export function movementAt(
  schedule: MovementSchedule | null | undefined,
  stemKey: string,
  timeSeconds: number,
  right = false,
): { position: [number, number, number]; gains: number[] } | undefined {
  const stem = movementStemFor(schedule, stemKey);
  const selected = stem && eventAt(stem.events, Math.max(0, timeSeconds) * 1_000_000);
  if (!selected) return undefined;
  const { current, previous } = selected;
  const currentPosition = right
    ? current.right_position ?? current.position
    : current.position;
  const previousPosition = right
    ? previous.right_position ?? previous.position
    : previous.position;
  const currentGains = right
    ? current.right_gains ?? current.gains
    : current.gains;
  const previousGains = right
    ? previous.right_gains ?? previous.gains
    : previous.gains;
  if (current === previous) return { position: currentPosition, gains: [...currentGains] };
  const amount = Math.min(
    1,
    Math.max(0, (Math.max(0, timeSeconds) * 1_000_000 - current.time_us) / current.interpolation_us),
  );
  return {
    position: interpolatePosition(previousPosition, currentPosition, amount),
    gains: interpolate(previousGains, currentGains, amount),
  };
}

/** ADM uses [x, -z, y], while the scene uses x/y/z with front at -z. */
export function scenePositionFromMovement(
  position: [number, number, number],
): { x: number; y: number; z: number } {
  return { x: position[0], y: position[2], z: -position[1] };
}
