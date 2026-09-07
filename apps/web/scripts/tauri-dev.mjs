import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const tauri = path.join(directory, "..", "node_modules", ".bin", "tauri");
const args = ["dev", ...process.argv.slice(2)];

if (process.platform === "darwin") {
  args.splice(
    1,
    0,
    "--runner",
    path.join(directory, "tauri-dev-cargo-runner.sh"),
  );
}

const child = spawn(tauri, args, { stdio: "inherit" });
child.on(
  "exit",
  (code, signal) => (process.exitCode = code ?? (signal ? 1 : 0)),
);
