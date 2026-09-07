import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const webDirectory = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);

test(
  "the development bundle contains the current executable, metadata, and resources",
  { skip: process.platform !== "darwin" },
  async () => {
    const directory = await mkdtemp(
      path.join(os.tmpdir(), "upmixer-dev-launcher-"),
    );
    const app = path.join(directory, "Upmixer.app");
    const binary = path.join(
      webDirectory,
      "src-tauri",
      "target",
      "debug",
      "upmixer-desktop",
    );
    try {
      execFileSync(
        process.execPath,
        ["scripts/tauri-dev-launcher.mjs", "--prepare", binary],
        {
          cwd: webDirectory,
          env: { ...process.env, UPMIXER_DEV_APP: app },
          stdio: "pipe",
        },
      );
      const executable = path.join(app, "Contents", "MacOS", "upmixer-desktop");
      assert.deepEqual(await readFile(executable), await readFile(binary));
      const plist = await readFile(
        path.join(app, "Contents", "Info.plist"),
        "utf8",
      );
      assert.match(plist, /com\.coderynx\.upmixer/);
      for (const resource of ["hrir", "xtc", "eq_fir"])
        assert.ok(
          (
            await stat(path.join(app, "Contents", "Resources", resource))
          ).isDirectory(),
        );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  },
);
