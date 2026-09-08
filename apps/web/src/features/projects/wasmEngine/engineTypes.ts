import { speakerCoordinates } from "@/lib/spatial";
import type {
  StemDynamicEqSettings,
  StemDynamicsSettings,
  StemEqSettings,
  StemMovementSettings,
} from "@/lib/manifest";

export type EngineRef<T> = { current: T };

export function engineRef<T>(value: T): EngineRef<T> {
  return { current: value };
}

export type OutputMode = "binaural" | "transaural" | "stereo" | "native" | "apple_spatial";

export const POSITIONAL_CHANNELS = Object.keys(speakerCoordinates);

export type MixPreview = {
  stem_routing?: Record<string, Record<string, number>>;
  bed_trim_db?: number;
  stem_rebalance?: Record<string, number>;
  stem_eq?: Record<string, string | StemEqSettings>;
  stem_dynamic_eq?: Record<string, StemDynamicEqSettings>;
  stem_dynamics?: Record<string, StemDynamicsSettings>;
  stem_movement?: Record<string, StemMovementSettings>;
  stem_ambient_rear?: Record<string, number>;
  stem_ambient_height?: Record<string, number>;
  stem_ambient_trim_db?: Record<string, number>;
  stem_height_texture?: Record<string, number>;
  stem_ambient_height_cutoff_hz?: Record<string, number>;
  stem_ambient_height_crossover_hz?: Record<string, number>;
  stem_object_mode?: Record<string, "linked-stereo" | "mono">;
  stem_placement?: Record<string, { azimuth_deg: number; elevation_deg: number; width_deg: number; object_size: number; diversity?: number; center_level_db?: number }>;
  stem_object_metadata?: Record<string, { gain?: number; importance?: number; channel_lock?: boolean; zone_exclusion?: string[] }>;
  spatial_downmix_lock?: boolean;
  stem_enabled?: Record<string, boolean>;
  stem_solo?: string[];
  stem_source_anchor_strength?: number;
};

/** One immutable event emitted by the shared movement compiler. */
export type MovementEvent = {
  time_us: number;
  position: [number, number, number];
  gains: number[];
  right_position?: [number, number, number] | null;
  right_gains?: number[] | null;
  interpolation_us: number;
};

export type MovementStemSchedule = {
  stem_key: string;
  stem_index: number;
  events: MovementEvent[];
};

/** Wire shape of the immutable schedule compiled off the audio thread. */
export type MovementSchedule = {
  version: number;
  revision: number;
  sample_rate: number;
  duration_frames: number;
  grid_us: number;
  interpolation_us: number;
  stems: MovementStemSchedule[];
};

/** The slow half of the master readout: what the delivered programme measures
 * and what it is being normalized to. Everything here is the *delivered*
 * value — the measurement plus whatever correction gain is applied — so it
 * reads against the target directly. */
export type LoudnessSummary = {
  /** BS.1770 integrated loudness, LKFS; -70 until the first pass lands. */
  integratedLkfs: number;
  /** Maximum true peak, dBTP. */
  truePeakDbtp: number;
  targetLkfs: number;
  ceilingDbtp: number;
  /** Monitor-only gain the loudness-matched A/B is applying, dB. Non-zero
   * only while the master chain is bypassed. */
  bypassMatchDb: number;
};

export const SILENT_LOUDNESS: LoudnessSummary = {
  integratedLkfs: -70,
  truePeakDbtp: -70,
  targetLkfs: -18,
  ceilingDbtp: -1,
  bypassMatchDb: 0,
};

export type EngineCallbacks = {
  onReady(ready: boolean): void;
  onLoadProgress(progress: number): void;
  onError(message: string | null): void;
  onPlaying(playing: boolean): void;
  onCurrentTime(time: number): void;
  onDuration(duration: number): void;
  onMeasuring(measuring: boolean): void;
  /** Measured loudness, the target it is normalized to, and the A/B match
   * gain — pushed whenever any of them moves, not per frame. */
  onLoudness(summary: LoudnessSummary): void;
  /** Fraction of the current measurement stage measured, for a progress bar. */
  onMeasureProgress(progress: number): void;
  onMaxChannels(maxChannels: number): void;
  onVolume(volume: number): void;
  onMuted(muted: boolean): void;
  onLoop(loop: boolean): void;
  onEngineStatus(kind: "native" | "wasm", fallbackReason: string | null): void;
  /** Installed movement schedule, for the scene and transport display. */
  onMovementSchedule?(schedule: MovementSchedule | null): void;
};
