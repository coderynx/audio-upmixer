import { FIELD_GRID, NumberField, SelectField, SliderField, SwitchRow } from "@/components/forms/fields";
import type { StemMovementRole, StemMovementSettings } from "@/lib/manifest";

export type StemMovementDefaults = Partial<StemMovementSettings>;

const STRUCTURAL_DEFAULTS: Pick<StemMovementSettings, "enabled" | "role" | "depth" | "start_s"> = {
  enabled: false,
  role: "auto",
  depth: 0,
  start_s: 0,
};

const ROLES = [
  { value: "auto", label: "Auto" },
  { value: "supporting", label: "Supporting" },
  { value: "featured", label: "Featured" },
];

const MOVEMENT_ANCHORS = new Set([
  "vocals", "lead vocals", "lead vocal", "bass", "kick", "snare", "drums", "combined drums",
]);

export function movementSettings(
  value: StemMovementSettings | undefined,
  defaults: StemMovementDefaults | undefined,
): StemMovementSettings | null {
  const response = value?.response ?? defaults?.response;
  const sensitivity = value?.sensitivity ?? defaults?.sensitivity;
  if (response == null || sensitivity == null) return null;
  return {
    ...STRUCTURAL_DEFAULTS,
    ...defaults,
    ...value,
    response,
    sensitivity,
  };
}

export function StemMovementControls({
  stemName,
  value,
  channels,
  defaults,
  onChange,
}: {
  stemName: string;
  value?: StemMovementSettings;
  channels: string[];
  defaults?: StemMovementDefaults;
  onChange: (value: StemMovementSettings) => void;
}) {
  if (MOVEMENT_ANCHORS.has(stemName.split("@", 1)[0].trim().toLowerCase())) return null;
  const movement = movementSettings(value, defaults);
  const stereo = channels.length === 2;
  if (!movement) return null;
  const disabled = stereo || !movement.enabled;
  const patch = (next: Partial<StemMovementSettings>) => onChange({ ...movement, ...next });

  return (
    <fieldset className="space-y-3 border-t pt-3" disabled={stereo}>
      <SwitchRow
        label="Movement"
        hint={stereo ? "Unavailable for ordinary stereo." : "Off freezes the Supporting placement."}
        checked={movement.enabled}
        disabled={stereo}
        onChange={(enabled) => patch({ enabled })}
      />
      {movement.enabled && (
        <>
          <SelectField
            label="Movement role"
            value={movement.role}
            disabled={disabled}
            onChange={(role) => patch({ role: role as StemMovementRole })}
            options={ROLES}
            hint="Supporting retains supporting motion; Featured approaches front-center while active."
          />
          <div className={FIELD_GRID}>
            <SliderField label="Movement depth" value={movement.depth} min={0} max={1} step={0.01} suffix="" disabled={disabled} onChange={(depth) => patch({ depth })} />
            <SliderField label="Movement response" value={movement.response} min={0.5} max={2} step={0.05} suffix="×" disabled={disabled} onChange={(response) => patch({ response })} />
            <SliderField label="Movement sensitivity" value={movement.sensitivity} min={0} max={1} step={0.01} disabled={disabled} onChange={(sensitivity) => patch({ sensitivity })} />
          </div>
          <div className={FIELD_GRID}>
            <NumberField label="Movement start" value={quantizeTime(movement.start_s)} min={0} step={0.02} suffix="s" disabled={disabled} onChange={(start_s) => patch({ start_s: quantizeTime(start_s ?? 0) })} />
            <NumberField label="Movement end" value={movement.end_s == null ? null : quantizeTime(movement.end_s)} min={0} step={0.02} suffix="s" hint="Leave empty to use the whole track." disabled={disabled} onChange={(end_s) => patch({ end_s: end_s == null ? null : quantizeTime(end_s) })} />
          </div>
        </>
      )}
    </fieldset>
  );
}

export function quantizeTime(seconds: number): number {
  return Math.max(0, Math.round(seconds / 0.02) * 0.02);
}
