import type { MovementSchedule } from "./engineTypes";
import type { StemMovementSettings as MovementSettings } from "@/lib/manifest";

export type CanonicalFeatures = {
  version: number;
  sample_rate: number;
  frame_count: number;
  window_frames: number;
  energies: number[];
};

export type MovementFeatureSidecar = {
  version: number;
  sample_rate: number;
  frame_count: number;
  stems: Array<{ stem_key: string; features: CanonicalFeatures }>;
};

export type MovementPlacement = {
  azimuth_deg: number;
  elevation_deg: number;
  width_deg: number;
  object_size: number;
  lfe: number;
  diversity: number;
  center_level_db: number;
};

export type MovementStemRequest = {
  stem_key: string;
  stem_name: string;
  /** Filled by the worker from the immutable feature sidecar. */
  features?: CanonicalFeatures;
  gain_db: number;
  enabled: boolean;
  included: boolean;
  placement: MovementPlacement;
  home_gains: number[];
  home_right_gains: number[];
  settings?: MovementSettings;
  object_mode?: "linked-stereo" | "mono" | null;
  channel_lock: boolean;
  zone_exclusion: string[];
};

export type MovementCompileRequest = {
  sample_rate: number;
  duration_frames: number;
  revision: number;
  channels: string[];
  stems: MovementStemRequest[];
  tuning?: Record<string, number>;
  /** The worker includes the validated source sidecar so the Rust boundary
   * can validate the complete identity/header contract before compiling. */
  feature_sidecar?: MovementFeatureSidecar;
};

type CompileMessage = {
  type: "compile";
  id: number;
  featuresUrl: string;
  /** Kept as metadata only: movement never extracts features from a proxy. */
  proxyUrl?: string;
  request: MovementCompileRequest;
  wasmModule?: WebAssembly.Module;
};

type WorkerResult = {
  type: "compiled";
  id: number;
  schedule: MovementSchedule;
} | {
  type: "error";
  id: number;
  message: string;
};

type WorkerLike = Pick<Worker, "postMessage" | "terminate"> & {
  onmessage: ((event: MessageEvent<WorkerResult>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
};

export type MovementWorkerFactory = () => WorkerLike;

const DEFAULT_WORKER: MovementWorkerFactory = () => new Worker(
  new URL("./movementWorkerRuntime.ts", import.meta.url),
  { type: "module" },
);

/** Main-thread handle for the dedicated movement compiler worker. The worker
 * owns feature fetching and the WASM instance; this class only tracks stale
 * revisions and resolves the latest result. */
export class MovementCompilerClient {
  private readonly worker: WorkerLike;
  private nextId = 0;
  private disposed = false;
  private pending = new Map<number, {
    revision: number;
    resolve: (schedule: MovementSchedule) => void;
    reject: (error: Error) => void;
  }>();

  constructor(
    workerFactory: MovementWorkerFactory = DEFAULT_WORKER,
    private readonly wasmModule?: WebAssembly.Module,
  ) {
    this.worker = workerFactory();
    this.worker.onmessage = (event) => this.onMessage(event.data);
    this.worker.onerror = (event) => {
      const error = new Error(event.message || "Movement compiler worker failed");
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
  }

  compile(
    featuresUrl: string,
    request: MovementCompileRequest,
    proxyUrl?: string,
  ): Promise<MovementSchedule> {
    if (this.disposed) return Promise.reject(new Error("Movement compiler is disposed"));
    const id = ++this.nextId;
    // A newer request supersedes every older one. Resolving stale requests to
    // rejection keeps callers from accidentally installing an old schedule.
    for (const [pendingId, pending] of this.pending) {
      if (request.revision > pending.revision) {
        pending.reject(new Error("stale movement compilation"));
        this.pending.delete(pendingId);
      }
    }
    return new Promise((resolve, reject) => {
      this.pending.set(id, { revision: request.revision, resolve, reject });
      const message: CompileMessage = {
        type: "compile",
        id,
        featuresUrl,
        proxyUrl,
        request,
        wasmModule: this.wasmModule,
      };
      this.worker.postMessage(message);
    });
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    const error = new Error("Movement compiler disposed");
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.worker.terminate();
  }

  private onMessage(message: WorkerResult) {
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.type === "error") {
      pending.reject(new Error(message.message));
      return;
    }
    if (message.schedule.revision !== pending.revision) {
      pending.reject(new Error("stale movement schedule"));
      return;
    }
    pending.resolve(message.schedule);
  }
}
